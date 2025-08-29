import numpy as np
import cv2
from collections import defaultdict
import json
import pytesseract
from PIL import Image
import io
import grading

BINARY_THRESHOLD = 150
BINARY_THRESHOLD_WARPED = 150
CONTOUR_EPSILON_FACTOR = 0.02
BORDER_MIN_AREA = 10000  # Adjust based on image resolution
RECTANGLE_MIN_WIDTH = 50
RECTANGLE_MIN_HEIGHT = 50
KERNEL_SIZE = (3, 3)
DILATION_ITERATIONS = 1
EROSION_ITERATIONS = 1
BLUR_KERNEL_SIZE = (5, 5)
BLUR_SIGMA = 0
BUBBLE_THRESHOLD = 127
BUBBLE_MIN_AREA = 50
BUBBLE_MAX_AREA = 500
BORDER_COLOR = (0, 255, 0)
BORDER_LINE_THICKNESS = 2
RECTANGLE_COLOR = (255, 0, 0)
RECTANGLE_LINE_THICKNESS = 2
BUBBLE_COLOR = (0, 0, 255)
BUBBLE_LINE_THICKNESS = 2

# Visualization colors
COLORS = {
    'hough_only': (255, 255, 0), # Cyan - Hough circles only
    'contour_only': (0, 255, 0), # Green - Contour filled only
    'both_methods': (0, 0, 255), # Red - Both methods agree (filled)
    'empty_bubble': (0, 165, 255), # Orange - Empty bubbles
    'all_bubbles': (255, 0, 0), # Blue - All detected bubbles
    'text': (255,255, 255), # White - Text labels
    'student_section': (255, 0, 255) # Magenta - Student number section
}

# Options mapping
OPTIONS = ['A', 'B', 'C', 'D', 'E']

# Student number section coordinates (adjust these based on your image)
STUDENT_SECTION = {
    'x':40,
    'y': 120,
    'w': 180,
    'h': 650,
    'name': 'Student Number',
    'format': ['digit', 'digit', 'letter', 'digit', 'digit', 'digit', 'digit']
}

# Options mapping for student number
DIGIT_OPTIONS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LETTER_OPTIONS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def find_borders(image):
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply binary threshold
    _, thresh = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    border_contours = []
    
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = CONTOUR_EPSILON_FACTOR * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If the approximated polygon has 4 vertices, it's likely a rectangle
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            # Look for larger rectangles that might be borders
            if area > BORDER_MIN_AREA:
                border_contours.append(contour)
    
    # If we found border contours, try to identify the corners
    if len(border_contours) >= 1:
        # Get the largest contour (assuming it's the main border)
        border_contour = max(border_contours, key=cv2.contourArea)
        
        # Simplify the contour to a quadrilateral
        epsilon = CONTOUR_EPSILON_FACTOR * cv2.arcLength(border_contour, True)
        approx = cv2.approxPolyDP(border_contour, epsilon, True)
        
        if len(approx) == 4:
            # Sort points: top-left, top-right, bottom-right, bottom-left
            points = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            
           
            # The bottom-right point will have the largest sum
            s = points.sum(axis=1)
            rect[0] = points[np.argmin(s)] # top-left
            rect[2] = points[np.argmax(s)] # bottom-right
            
            # The top-right point will have the smallest difference,
            # The bottom-left will have the largest difference
            diff = np.diff(points, axis=1)
            rect[1] = points[np.argmin(diff)] # top-right
            rect[3] = points[np.argmax(diff)] # bottom-left
            
            return rect
    
    # if we can't find borders return the image boundaries
    height, width = image.shape[:2]
    return np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

def perspective_transform(image, corners):
    # Order points: top-left, top-right, bottom-right, bottom-left
    (tl, tr, br, bl) = corners
    
    # Compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Construct destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(corners, dst)
    
    # Apply the perspective transform
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def auto_rotate_image(image):
   
    try:
        # Convert to PIL Image for processing
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Get orientation and script detection
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        osd = pytesseract.image_to_osd(pil_image, output_type=pytesseract.Output.DICT)
        
        print(f"Detected orientation: {osd['orientation']} degrees")
        print(f"Detected script: {osd['script']}")
        print(f"Orientation confidence: {osd['orientation_conf']}")
        
        # Rotate image if needed
        if osd['orientation'] != 0 and osd['orientation_conf'] > 0.1:
            print("ROTATE")
            # Calculate rotation angle
            rotation_angle = osd['orientation']
            
            # Rotate the image
            rotated = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).rotate(
                rotation_angle, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255)
            )
            
            # Convert back to OpenCV format
            rotated_cv = cv2.cvtColor(np.array(rotated), cv2.COLOR_RGB2BGR)
            print(f"Image rotated by {rotation_angle} degrees")
            return rotated_cv
        elif osd['orientation_conf'] < 0.1:
            rotated_cv = cv2.rotate(image, cv2.ROTATE_180)
            return rotated_cv
        else:
            print("No rotation needed")
            return image
            
    except Exception as e:
        print(f"Auto-rotation failed: {e}")
        return image

def load_image(image_path):
    
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Apply auto-rotation
    rotated_img = auto_rotate_image(original_img)
    
    # Apply perspective correction
    print("Finding borders...")
    borders = find_borders(rotated_img)
    
    print("Applying perspective correction...")
    warped = perspective_transform(rotated_img, borders)
    
    img = warped.copy()
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    return rotated_img, img, gray  # rotated_img is pre-warp for original display

def preprocess_image(gray):
    
    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Optional morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    
    return blurred, opened

def detect_hough_circles(gray):
   
    blurred, _ = preprocess_image(gray)
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.01,
        minDist=4,
        param1=40,
        param2=20,
        minRadius=8,
        maxRadius=10
    )
    
    hough_bubbles = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            hough_bubbles.append({
                'center': (x, y),
                'radius': r,
                'method': 'hough'
            })
            
    print(f"Hough Circles detected: {len(hough_bubbles)} bubbles")
    return hough_bubbles

def detect_contour_filled(gray, section_bounds):
    
    contour_filled = []
    
    x, y, w, h = section_bounds['x'], section_bounds['y'], section_bounds['w'], section_bounds['h']
    
    # Get ROI for this section
    section_roi = gray[y:y + h, x:x + w]
    section_roi_blurred = cv2.GaussianBlur(section_roi, (5, 5), 0)
    
    # Apply threshold to find filled areas
    ret, thresh = cv2.threshold(section_roi_blurred, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find filled centers in this section
    for shape in contours:
        area = cv2.contourArea(shape)
        if 210 < area < 400: # Filled bubble size range
            (cx, cy), radius = cv2.minEnclosingCircle(shape)
            cx_full = int(cx) + x
            cy_full = int(cy) + y
            
            contour_filled.append({
                'center': (cx_full, cy_full),
                'radius': int(radius),
                'method': 'contour',
                'section': section_bounds['name']
            })
    
    print(f"Contour analysis detected: {len(contour_filled)} filled areas in {section_bounds['name']}")
    return contour_filled

def combine_detections(hough_bubbles, contour_filled, tolerance=5):
   
    combined_bubbles = []
    matched_contours = set()
    
    # For each Hough circle, check if there's a matching contour
    for hough_bubble in hough_bubbles:
        hx, hy = hough_bubble['center']
        hr = hough_bubble['radius']
        
        best_match = None
        best_distance = float('inf')
        best_idx = -1
        
        # Find closest contour detection
        for idx, contour_bubble in enumerate(contour_filled):
            if idx in matched_contours:
                continue
                
            cx, cy = contour_bubble['center']
            distance = np.sqrt((hx - cx)**2 + (hy - cy)**2)
            
            if distance < tolerance and distance < best_distance:
                best_match = contour_bubble
                best_distance = distance
                best_idx = idx
        
        if best_match:
            # Both methods agree - this bubble is filled
            combined_bubbles.append({
                'center': hough_bubble['center'],
                'radius': hough_bubble['radius'],
                'status': 'filled_confirmed',
                'methods': ['hough', 'contour'],
                'confidence': 'high'
            })
            matched_contours.add(best_idx)
        else:
            # Only Hough detected - likely empty bubble
            combined_bubbles.append({
                'center': hough_bubble['center'],
                'radius': hough_bubble['radius'],
                'status': 'empty',
                'methods': ['hough'],
                'confidence': 'medium'
            })
    
    # Add unmatched contour detections (contour-only filled areas)
    for idx, contour_bubble in enumerate(contour_filled):
        if idx not in matched_contours:
            combined_bubbles.append({
                'center': contour_bubble['center'],
                'radius': contour_bubble['radius'],
                'status': 'filled_contour_only',
                'methods': ['contour'],
                'confidence': 'medium'
            })
    
    return combined_bubbles

def organize_into_questions(combined_bubbles, column_bounds):
   
    all_answers = {}
    question_details = {}
    
    for col_idx, column in enumerate(column_bounds):
        x, y, w, h = column['x'], column['y'], column['w'], column['h']
        
        # Find bubbles in this column
        column_bubbles = []
        for bubble in combined_bubbles:
            bx, by = bubble['center']
            if x <= bx <= x + w and y <= by <= y + h:
                column_bubbles.append(bubble)
        
        # Sort bubbles by position (top to bottom, left to right)
        column_bubbles.sort(key=lambda b: (b['center'][1], b['center'][0]))
        
        # Group into rows of 5 (A, B, C, D, E)
        question_rows = []
        for i in range(0, len(column_bubbles), 5):
            row = column_bubbles[i:i+5]
            if len(row) == 5:
                # Sort row by x-coordinate for A-B-C-D-E order
                row.sort(key=lambda b: b['center'][0])
                question_rows.append(row)
        
        # Process each question
        for row_idx, row_bubbles in enumerate(question_rows):
            question_num = (row_idx + 1) if col_idx == 0 else (row_idx + 31)
            
            filled_options = []
            question_detail = {
                'bubbles': [],
                'filled_options': [],
                'confidence_scores': [],
                'detection_methods': []
            }
            
            # Check each option (A, B, C, D, E)
            for bubble_idx, bubble in enumerate(row_bubbles):
                option_letter = OPTIONS[bubble_idx]
                is_filled = bubble['status'] in ['filled_confirmed', 'filled_contour_only']
                
                question_detail['bubbles'].append({
                    'option': option_letter,
                    'center': bubble['center'],
                    'radius': bubble['radius'],
                    'status': bubble['status'],
                    'methods': bubble['methods'],
                    'confidence': bubble['confidence'],
                    'filled': is_filled
                })
                
                if is_filled:
                    filled_options.append(option_letter)
                    question_detail['filled_options'].append(option_letter)
                    question_detail['confidence_scores'].append(bubble['confidence'])
                    question_detail['detection_methods'].append(bubble['methods'])
            
            all_answers[question_num] = filled_options
            question_details[question_num] = question_detail
    
    return all_answers, question_details

def organize_student_number(combined_bubbles, student_section):
    
    x, y, w, h = student_section['x'], student_section['y'], student_section['w'], student_section['h']
    
   
    student_bubbles = []
    for bubble in combined_bubbles:
        bx, by = bubble['center']
        if x <= bx <= x + w and y <= by <= y + h:
            student_bubbles.append(bubble)
    
    # Sort bubbles by x-coordinate (left to right) to group into columns
    student_bubbles.sort(key=lambda b: b['center'][0])
    
   
    columns = defaultdict(list)
    for bubble in student_bubbles:
        bx = bubble['center'][0]
        # Group bubbles that are close in x-coordinate (same column)
        found_col = False
        for col_x in columns.keys():
            if abs(bx - col_x) < 15: # Tolerance for column grouping
                columns[col_x].append(bubble)
                found_col = True
                break
        if not found_col:
            columns[bx] = [bubble]
    
   
    sorted_columns = sorted(columns.items(), key=lambda x: x[0])
    
    # Process each column to determine the student number
    student_number = []
    student_details = {}
    
    for i, (col_x, bubbles) in enumerate(sorted_columns):
        if i >= len(student_section['format']):
            break
            
      
        bubbles.sort(key=lambda b: b['center'][1])
        
        # Determine the type of options for this position
        if student_section['format'][i] == 'digit':
            options = DIGIT_OPTIONS
        else: # letter
            options = LETTER_OPTIONS
        
        # Find the filled bubble in this column
        filled_option = None
        col_details = {
            'position': i + 1,
            'type': student_section['format'][i],
            'bubbles': []
        }
        
        for j, bubble in enumerate(bubbles):
            if j >= len(options):
                break
                
            option = options[j]
            is_filled = bubble['status'] in ['filled_confirmed', 'filled_contour_only']
            
            col_details['bubbles'].append({
                'option': option,
                'center': bubble['center'],
                'radius': bubble['radius'],
                'status': bubble['status'],
                'methods': bubble['methods'],
                'confidence': bubble['confidence'],
                'filled': is_filled
            })
            
            if is_filled:
                filled_option = option
        
        student_details[i + 1] = col_details
        student_number.append(filled_option if filled_option else '?')
    
    print(f"Organized {len(student_bubbles)} student number bubbles into {len(student_number)} positions")
    return student_number, student_details

def visualize_results(img, combined_bubbles, all_answers, question_details, student_section=None, student_number=None, student_details=None):
    """Draw visualization with color coding for different detection types and student number labels"""
    
    # Draw all bubbles with appropriate colors
    for bubble in combined_bubbles:
        center = bubble['center']
        radius = bubble['radius']
        status = bubble['status']
        
        if status == 'filled_confirmed':
            color = COLORS['both_methods']
            thickness = 3
        elif status == 'filled_contour_only':
            color = COLORS['contour_only']
            thickness = 2
        elif status == 'empty':
            color = COLORS['empty_bubble']
            thickness = 1
        else:
            color = COLORS['hough_only']
            thickness = 2
            
        cv2.circle(img, center, radius, color, thickness)
    
    # Add option labels for questions
    for q_num, details in question_details.items():
        for bubble_info in details['bubbles']:
            center = bubble_info['center']
            option = bubble_info['option']
            cv2.putText(img, option, (center[0] - 6, center[1] + 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text'], 1)
    
    # Draw student number section and labels
    if student_section and student_details:
        x, y, w, h = student_section['x'], student_section['y'], student_section['w'], student_section['h']
        cv2.rectangle(img, (x, y), (x + w, y + h), COLORS['student_section'], 3)
        cv2.putText(img, student_section['name'], (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['student_section'], 2)
        
        for pos, details in student_details.items():
            for bubble_info in details['bubbles']:
                center = bubble_info['center']
                option = bubble_info['option']
                cv2.putText(img, option, (center[0] - 6, center[1] + 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text'], 1)
    
    # Display detected student number
    if student_number:
        student_str = ''.join(student_number)
        
def generate_report(all_answers, question_details, hough_bubbles, contour_filled, student_number=None, student_details=None):
    """Generate detailed analysis report with student number in specified format"""
    print("\n" + "="*80)
    print("ENHANCED MCQ BUBBLE DETECTION ANALYSIS")
    print("="*80)
    
    # Method statistics
    hough_count = len(hough_bubbles)
    contour_count = len(contour_filled)
    
    both_methods_count = 0
    contour_only_count = 0
    empty_count = 0
    
    total_filled = 0
    high_confidence_filled = 0
    
    for details in question_details.values():
        for bubble in details['bubbles']:
            if bubble['status'] == 'filled_confirmed':
                both_methods_count += 1
                total_filled += 1
                high_confidence_filled += 1
            elif bubble['status'] == 'filled_contour_only':
                contour_only_count += 1
                total_filled += 1
            elif bubble['status'] == 'empty':
                empty_count += 1
    
    print(f"\nDETECTION METHOD STATISTICS:")
    
    print(f" Hough Circles detected: {hough_count} total bubbles")
    print(f" Contour analysis found: {contour_count} filled areas")
    print(f" Both methods agreed: {both_methods_count} (HIGH confidence)")
    print(f" Contour only: {contour_only_count} (MEDIUM confidence)")
    print(f" Empty bubbles: {empty_count}")
    print(f" Total filled bubbles: {total_filled}")
    
    agreement_rate = 0
    if hough_count > 0:
        agreement_rate = (both_methods_count / hough_count) * 100
        print(f" Method agreement rate: {agreement_rate:.1f}%")
    
    # Student number section
    if student_number and student_details:
        print(f"\nSTUDENT NUMBER SECTION:")
        student_number_str = ''.join(student_number)
        print(f" Detected Student Number: {student_number_str}")
        
        print(f"\nDETAILED BREAKDOWN:")
        print("-" * 50)
        
        for pos, details in student_details.items():
            pos_type = "Digit" if details['type'] == 'digit' else "Letter"
            filled_options = []
            
            for bubble in details['bubbles']:
                if bubble['filled']:
                    filled_options.append(bubble['option'])
            
            if len(filled_options) == 1:
                status = filled_options[0]
            elif len(filled_options) > 1:
                status = f"{', '.join(filled_options)}  MULTIPLE SELECTIONS FOUND"
            else:
                status = "UNANSWERED"
            
            print(f"Position {pos} ({pos_type}): {status}")
            
            # Show detection details for filled bubbles
            for bubble in details['bubbles']:
                if bubble['filled']:
                    methods_str = "+".join(bubble['methods'])
                    confidence_str = bubble['confidence'].upper()
                    print(f" {bubble['option']}: {methods_str} ({confidence_str})")
    
    # Question analysis
    answered_questions = 0
    multiple_selections = 0
    unanswered_questions = 0
    high_confidence_answers = 0
    
    print(f"\nQUESTION-BY-QUESTION BREAKDOWN:")
    print("-" * 50)
    
    for q_num in sorted(all_answers.keys()):
        answers = all_answers[q_num]
        details = question_details[q_num]
        
        if len(answers) == 0:
            status = "[UNANSWERED]"
            unanswered_questions += 1
        elif len(answers) == 1:
            status = answers[0]
            answered_questions += 1
            for bubble in details['bubbles']:
                if bubble['option'] == answers[0] and bubble['confidence'] == 'high':
                    high_confidence_answers += 1
                    break
        else:
            status = f"{', '.join(answers)} ← MULTIPLE SELECTIONS"
            multiple_selections += 1
            answered_questions += 1
        
        detection_info = ""
        for bubble in details['bubbles']:
            if bubble['filled']:
                methods_str = "+".join(bubble['methods'])
                confidence_str = bubble['confidence'].upper()
                detection_info += f" [{bubble['option']}:{methods_str}({confidence_str})]"
        
        print(f"Question {q_num:2d}: {status:15s} {detection_info}")
    
    # Summary statistics
    total_questions = len(all_answers)
    
    print(f"\nSUMMARY STATISTICS:")
    print("-" * 30)
    print(f"Total Questions: {total_questions}")
    print(f"Answered: {answered_questions}")
    print(f" - High Confidence: {high_confidence_answers}")
    print(f" - Medium Confidence: {answered_questions - high_confidence_answers}")
    print(f"Unanswered: {unanswered_questions}")
    print(f"Multiple Selections: {multiple_selections}")
    
    if total_questions > 0:
        completion_rate = (answered_questions / total_questions) * 100
        confidence_rate = (high_confidence_answers / answered_questions) * 100 if answered_questions > 0 else 0
        
        print(f"Completion Rate: {completion_rate:.1f}%")
        
    print(f"\nRECOMMENDATIONS:")
    print("-" * 20)
    
    if multiple_selections > 0:
        print(f"{multiple_selections} questions have multiple selections - review needed")
    
    if contour_only_count > both_methods_count * 0.3:
        print("High number of contour-only detections - consider adjusting Hough parameters")
    
    if high_confidence_answers < answered_questions * 0.8:
        print(" Low high-confidence detection rate - consider image quality improvement")
    
    result = {
        'total_questions': total_questions,
        'answered': answered_questions,
        'high_confidence': high_confidence_answers,
        'multiple_selections': multiple_selections,
        'method_agreement_rate': agreement_rate
    }
    
    if student_number and student_details:
        result['student_digits_detected'] = sum(1 for char in student_number if char != '?')
        result['student_number'] = ''.join(student_number)
    
    return result

def process_mcq_with_student_section(image_path):

    original_img, img, gray = load_image(image_path)
    
  
    height, width = gray.shape
    column_bounds = [
        { # Left column
            'x': 280,
            'y': 30,
            'w': width // 2 - 200,
            'h': height - 80,
            'name': 'Left Column (Q1-30)'
        },
        { # Right column
            'x': width // 2 + 140,
            'y': 30,
            'w': width // 2 - 200,
            'h': height - 80,
            'name': 'Right Column (Q31-60)'
        }
    ]
    
    # Detection pipeline
    print("Detecting bubbles with Hough Circles...")
    hough_bubbles = detect_hough_circles(gray)
    
    print("Detecting filled areas with contour analysis...")
    contour_filled_answers = detect_contour_filled(gray, column_bounds[0])
    contour_filled_answers.extend(detect_contour_filled(gray, column_bounds[1]))
    
    # Detect in student number section
    contour_filled_student = detect_contour_filled(gray, STUDENT_SECTION)
    
    # Combine all contour detections
    contour_filled = contour_filled_answers + contour_filled_student
    
    print("Combining detection methods...")
    combined_bubbles = combine_detections(hough_bubbles, contour_filled)
    
    print("Organizing into questions...")
    all_answers, question_details = organize_into_questions(combined_bubbles, column_bounds)
    
    # Organize student number
    student_number, student_details = organize_student_number(combined_bubbles, STUDENT_SECTION)
    
    print("Generating visualization and report...")
    visualize_results(img, combined_bubbles, all_answers, question_details, STUDENT_SECTION, student_number, student_details)
   
    for column in column_bounds:
        x, y, w, h = column['x'], column['y'], column['w'], column['h']
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, column['name'], (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    stats = generate_report(all_answers, question_details, hough_bubbles, contour_filled, student_number, student_details)
    
    return original_img, img, all_answers, question_details, stats, student_number, student_details

def display_results(original_img, processed_img):
    """Display the analyzed image"""
    cv2.imshow('Original', original_img)
    cv2.imshow('Enhanced MCQ Analysis', processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def export_results(answers, student_number=None, filename='mcq_results.json'):
   
    results_export = {
        "answers": answers
    }
    if student_number:
        results_export["student_number"] = student_number
    
    with open(filename, "w") as f:
        json.dump(results_export, f, indent=2)
    print(f"Results exported to {filename}")

import os
import glob


def process_all_images_in_folder(folder_path):
   
   
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, extension)))
    
    print(f"Found {len(image_files)} images in folder: {folder_path}")
    
    
    for image_path in image_files:
        try:
            print(f"\n{'='*60}")
            print(f"PROCESSING: {os.path.basename(image_path)}")
            print(f"{'='*60}")
            
            
            original_img, processed_img, answers, details, statistics, student_number, student_details = process_mcq_with_student_section(image_path)
            
            
            filename_base = os.path.splitext(os.path.basename(image_path))[0]
            export_filename = f"mcq_results_{filename_base}.json"
            export_results(answers, student_number, export_filename)
            result= grading.grade_student(export_filename)
            grading.export_to_csv(result)
            grading.sort_csv_and_calculate_average()
            print(f"✅ Processed and exported: {export_filename}")
            
            
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")

if __name__ == "__main__":
    folder_path = 'Solutions/MCQ_600dpi_2016/'
    
    # Process all images in the folder
    process_all_images_in_folder(folder_path)
    
    print("\n🎉 All images processed!")