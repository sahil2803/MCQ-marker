import json
import csv
import os

# -----------------------------
# Configuration
# -----------------------------
RESULTS_FILE = 'mcq_results.json'
ANSWER_KEY_FILE = 'answer_key_weighted.json'
CSV_OUTPUT = 'grading_results.csv'

def ui():
    print("GRADING SYSTEM")

def grade_student(json_file='mcq_results.json', key_file='answer_key_weighted.json'):
    # Check file exists
    if not os.path.exists(json_file):
        print(f"❌ Error: '{json_file}' not found!")
        return None

    if os.path.getsize(json_file) == 0:
        print(f"❌ Error: '{json_file}' is empty!")
        return None

    # Load student results
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in '{json_file}': {e}")
        return None

    # Load answer key
    try:
        with open(key_file, 'r', encoding='utf-8') as f:
            answer_key = json.load(f)
    except Exception as e:
        print(f"❌ Could not load answer key: {e}")
        return None

    # Extract student answers and ID
    student_answers = data.get('answers', {})
    student_number = ''.join(data.get('student_number', ['UNKNOWN']))

    # Initialize
    total_score = 0.0
    total_possible = 0.0
    correct_count = 0
    incorrect_count = 0
    unanswered_count = 0
    feedback = []

    # Grade each question
    for q_str, info in answer_key.items():
        q_num = int(q_str)
        correct = info['correct_answer']
        weight = float(info.get('weight', 1.0))
        total_possible += weight

        # Check if student answered
        if str(q_num) not in student_answers:
            feedback.append(f"Q{q_num}: ❌ (Missing in data)")
            unanswered_count += 1
            continue

        student_ans = student_answers[str(q_num)]

        # Unanswered
        if len(student_ans) == 0:
            feedback.append(f"Q{q_num}: ❌ (Unanswered) [{weight}pt]")
            unanswered_count += 1
        # Single answer and correct
        elif len(student_ans) == 1 and student_ans[0] == correct:
            total_score += weight
            correct_count += 1
            feedback.append(f"Q{q_num}: ✅ {correct} [{weight}pt]")
        # Multiple selections or wrong
        else:
            incorrect_count += 1
            given = ''.join(student_ans)
            feedback.append(f"Q{q_num}: ❌ {given} → {correct} [{weight}pt]")

    # Final score
    percentage = (total_score / total_possible * 100) if total_possible > 0 else 0

    # Print Report
    print("\n" + "=" * 60)
    print("📝 STUDENT MCQ GRADING REPORT (WEIGHTED)")
    print("=" * 60)
    print(f"Student Number: {student_number}")
    print(f"Total Possible Marks: {total_possible}")
    print(f"Marks Scored: {total_score:.1f}")
    print(f"Score: {percentage:.1f}%")
    print(f"Correct: {correct_count} | Incorrect: {incorrect_count} | Unanswered: {unanswered_count}")
    print("-" * 60)

    for line in feedback:
        print(line)

    print("=" * 60)

    return {
        'student_number': student_number,
        'total_possible': total_possible,
        'marks_obtained': round(total_score, 2),
        'percentage': round(percentage, 1),
        'correct': correct_count,
        'incorrect': incorrect_count,
        'unanswered': unanswered_count
    }

def export_to_csv(report, csv_file='grading_results.csv'):
    fieldnames = ['student_number', 'total_possible', 'marks_obtained',
                  'percentage', 'correct', 'incorrect', 'unanswered']
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(report)

    print(f"📊 Result saved to {csv_file}")

def sort_csv_and_calculate_average(csv_file='grading_results.csv'):
    """
    Sort CSV by highest to lowest marks and calculate average mark
    """
    if not os.path.exists(csv_file):
        print(f"❌ CSV file '{csv_file}' not found!")
        return
    
    try:
        # Read all data from CSV
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            print("❌ No data found in CSV file!")
            return
        
        # Sort by percentage (highest to lowest)
        sorted_rows = sorted(rows, key=lambda x: float(x['percentage']), reverse=True)
        
        # Calculate average
        total_percentage = sum(float(row['percentage']) for row in sorted_rows)
        average_percentage = total_percentage / len(sorted_rows)
        
        total_marks = sum(float(row['marks_obtained']) for row in sorted_rows)
        average_marks = total_marks / len(sorted_rows)
        
        # Write sorted data back to CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(sorted_rows)
        
        print("\n" + "=" * 60)
        print("📊 CLASS RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total Students: {len(sorted_rows)}")
        print(f"Average Marks: {average_marks:.2f}/{float(sorted_rows[0]['total_possible']):.1f}")
        print(f"Average Percentage: {average_percentage:.1f}%")
        print("\n🏆 TOP 5 STUDENTS:")
        print("-" * 40)
        
        for i, row in enumerate(sorted_rows[:5], 1):
            print(f"{i}. {row['student_number']}: {row['percentage']}% ({row['marks_obtained']}/{row['total_possible']})")
        
        print("\n📈 FULL RANKING (Saved to CSV):")
        print("-" * 40)
        for i, row in enumerate(sorted_rows, 1):
            print(f"{i:2d}. {row['student_number']}: {row['percentage']}%")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error processing CSV: {e}")

def email():
    print("Email Students")

# -----------------------------
# Run Grading
# -----------------------------
if __name__ == "__main__":
    # Grade the student
    result = grade_student('mcq_results.json', 'answer_key_weighted.json')
    
    if result:
        # Export to CSV
        export_to_csv(result)
        
        # Sort CSV and show class statistics
        sort_csv_and_calculate_average()