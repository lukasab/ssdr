"""Finds the values in the screen of input multimeter photo"""
import argparse
from datetime import datetime
import cv2
import numpy as np
from imutils import contours, is_cv2
from imutils.perspective import four_point_transform

PARSER = argparse.ArgumentParser()
PARSER.add_argument(
    "-i", "--image", help="the input image file.", required=True
)

PARSER.add_argument(
    "-lrc",
    "--lowRangeColor",
    help="the low range values for hue, sat and val.",
    nargs="+",
    default=[21, 14, 125],
    type=int,
)

PARSER.add_argument(
    "-hrc",
    "--highRangeColor",
    help="the high range values for hue, sat and val.",
    nargs="+",
    default=[79, 63, 185],
    type=int,
)

PARSER.add_argument(
    "-yTB",
    "--yTopBot",
    help="Top to bottom porcentage to crop from the image.",
    default=0.1,
    type=int,
)

PARSER.add_argument(
    "-yBT",
    "--yBotTop",
    help="Bottom to top porcentage to crop from the image.",
    default=0.8,
    type=int,
)

PARSER.add_argument(
    "-xLR",
    "--xLeftRight",
    help="Left to right porcentage to crop from the image.",
    default=0.035,
    type=int,
)

PARSER.add_argument(
    "-xRL",
    "--xRightLeft",
    help="Right to left porcentage to crop from the image.",
    default=0.87,
    type=int,
)

PARSER.add_argument(
    "-rf",
    "--resizeFactor",
    help="Resize Factor to use on image.",
    default=1,
    type=int,
)

PARSER.add_argument(
    "-d",
    "--debug",
    help="Debug mode? True or False",
    default="False",
    type=str,
)

PARSER.add_argument(
    "-sp",
    "--screenPoints",
    help="Want to give screen points position? True or False\
          [top_left, top_right, bottom_right, bottom_left]",
    default="False",
    type=str,
)

ARGS = vars(PARSER.parse_args())
HSVL = np.array(ARGS["lowRangeColor"], np.uint8)
HSVH = np.array(ARGS["highRangeColor"], np.uint8)
YTB = ARGS["yTopBot"]
YBT = ARGS["yBotTop"]
XLR = ARGS["xLeftRight"]
XRL = ARGS["xRightLeft"]
RFACTOR = ARGS["resizeFactor"]

DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9,
    (0, 0, 0, 0, 0, 0, 0): "",
}

SCREEN_AREA_THRESHOLD = 100000
DIST_THRESHOLD = 5.7
SEGMENT_AREA_THRESHOLD = 764
HSVL_LIST = [
    np.array(ARGS["lowRangeColor"], np.uint8),
    [53, 76, 112],
    [52, 25, 120],
    [86, 80, 60],
]
HSVH_LIST = [
    np.array(ARGS["highRangeColor"], np.uint8),
    [103, 118, 235],
    [102, 80, 235],
    [98, 189, 129],
]


def str2bool(argument_string):
    """Transform string to boolean value"""
    if argument_string.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if argument_string.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def color_screen_threshold(img, hsvl, hsvh):
    """Create a mask with the HSV plane"""
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    screen_mask = cv2.inRange(
        hsv_img, np.array(hsvl, np.uint8), np.array(hsvh, np.uint8)
    )
    kernel = np.ones((2, 2), np.uint8)
    screen_filter = cv2.dilate(screen_mask, kernel)
    if DEBUG:
        cv2.imshow("hsv", hsv_img)
        cv2.imshow("mask", screen_mask)
        cv2.imshow("filtered", screen_filter)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return screen_filter


def get_screen_area(screen_filtered):
    """Get the screen area, should be the biggest contour Area"""
    screen_cnts = cv2.findContours(
        screen_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    screen_cnts = screen_cnts[0] if is_cv2(or_better=True) else screen_cnts[1]
    try:
        biggest_cnt = sorted(screen_cnts, key=cv2.contourArea, reverse=True)[
            0
        ]
        screen_area = cv2.contourArea(biggest_cnt)
        return (biggest_cnt, screen_area)
    except IndexError:
        return (False, 0)


def find_screen(img_res):
    """Tries to find the multimeter screen based on its color,
       if not possible use default position"""
    screen_area = 0
    for HSVL, HSVJ in zip(HSVL_LIST, HSVH_LIST):
        screen_filtered = color_screen_threshold(img_res, HSVL, HSVH)
        (biggest_cnt, screen_area) = get_screen_area(screen_filtered)
        if screen_area >= SCREEN_AREA_THRESHOLD:
            screen_box = np.int0(cv2.boxPoints(cv2.minAreaRect(biggest_cnt)))
            break
    else:
        x_screen = img_res.shape[1]
        y_screen = img_res.shape[0]
        top_left = (round(x_screen / 4), round(y_screen / 4))
        top_right = (round(3 * x_screen / 4), round(y_screen / 4))
        bottom_right = (round(3 * x_screen / 4), round(3 * y_screen / 4))
        bottom_left = (round(x_screen / 4), round(3 * y_screen / 4))
        screen_box = np.array(
            [top_left, top_right, bottom_right, bottom_left],
            dtype="float32",
        )
    return four_point_transform(img_res, np.reshape(screen_box, (4, 2)))


def open_resize_image(resive_factor):
    """Opens image and resive it to given factor"""
    return cv2.resize(
        cv2.imread(ARGS["image"], 1),
        (0, 0),
        fx=resive_factor,
        fy=resive_factor,
    )


def filter_screen(
    screen,
    kernel_size,
    erosion_iteration,
    dilation_iteration,
    operation="opening",
):
    """Filter screen with a opening or closing morphological operation"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if operation == "opening":
        erosion = cv2.erode(
            screen.copy(), kernel, iterations=erosion_iteration
        )
        return cv2.dilate(erosion, kernel, iterations=dilation_iteration)

    dilation = cv2.dilate(
        screen.copy(), kernel, iterations=dilation_iteration
    )
    return cv2.erode(dilation, kernel, iterations=erosion_iteration)


def cropp_to_digits(image):
    """Finds the first pixel at the top and bottom and crops the image"""
    stop = False
    height, width = image.shape
    for y_pixel in range(0, height):
        for x_pixel in range(0, width):
            if image[y_pixel, x_pixel] == 0:
                first_pixel_up = y_pixel
                stop = True
                break
        if stop is True:
            break

    stop = False
    for y_pixel in range(height - 1, -1, -1):
        for x_pixel in range(0, width):
            if image[y_pixel, x_pixel] == 0:
                first_pixel_down = y_pixel
                stop = True
                break
        if stop is True:
            break

    return image[first_pixel_up : first_pixel_down + 1]


def find_if_close(cnt1, cnt2, dist_th):
    """Check if contours should be joined, segments sometimes separate"""
    row1, row2 = cnt1.shape[0], cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            (x_bottom_1, _, _, _) = cv2.boundingRect(cnt1[i])
            (x_bottom_2, _, _, _) = cv2.boundingRect(cnt2[j])
            if abs(x_bottom_1 - x_bottom_2) == 0:
                return True
            dist = np.linalg.norm(cnt1[i] - cnt2[j])
            if dist < dist_th:
                return True
            if i == row1 - 1 and j == row2 - 1:
                return False
    return False


def join_cnts(image, sorted_contours, dist_th=DIST_THRESHOLD):
    """Join contours that represent one digit"""
    status = np.zeros((len(sorted_contours), 1))

    for i, cnt1 in enumerate(sorted_contours):
        x_counter = i
        if i != len(sorted_contours) - 1:
            for _, cnt2 in enumerate(sorted_contours[i + 1 :]):
                x_counter = x_counter + 1
                dist = find_if_close(cnt1, cnt2, dist_th)
                if dist:
                    val = min(status[i], status[x_counter])
                    status[x_counter] = status[i] = val
                else:
                    if status[x_counter] == status[i]:
                        status[x_counter] = i + 1
    unified = []
    maximum = int(status.max()) + 1
    for i in range(maximum):
        pos = np.where(status == i)[0]
        if pos.size != 0:
            cont = np.vstack([sorted_contours[i] for i in pos])
            hull = cv2.convexHull(cont)
            unified.append(hull)

    image_color2 = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    if DEBUG:
        cv2.drawContours(image_color2, unified, -1, (0, 255, 0), 2)
        cv2.imshow("image", image_color2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return contours.sort_contours(unified)[0]


def find_digits_cnts(image, dist_th=DIST_THRESHOLD):
    """Returns all digits contours in image, sorted from left to right"""
    screen_digit_cnts = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    screen_digit_cnts = (
        screen_digit_cnts[0]
        if is_cv2(or_better=True)
        else screen_digit_cnts[1]
    )
    y_image = image.shape[0]
    digits_cnts = []
    for cnt in screen_digit_cnts:
        (_, y_bottom, width, height) = cv2.boundingRect(cnt)
        if (width * height > SEGMENT_AREA_THRESHOLD) or (
            width * height > int(SEGMENT_AREA_THRESHOLD / 2) and width > 50
        ):
            digits_cnts.append(cnt)

    image_color2 = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    if DEBUG:
        cv2.drawContours(image_color2, digits_cnts, -1, (0, 0, 255), 1)
        cv2.imshow("image2", image_color2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    sorted_contours = contours.sort_contours(digits_cnts)[0]

    return join_cnts(image, sorted_contours, dist_th)


def check_segments(roi, point_side="None"):
    """Check if each segment is on or off"""
    (roi_height, roi_width) = roi.shape
    (d_width, d_height) = (int(roi_width * 0.24), int(roi_height * 0.11))
    d_height_center = int(d_height * 0.5)
    d_width_center = int(d_width * 0.5)

    # top, top-left, top-right, center, bottom_left, bottom_right, bottom
    if point_side == "None":
        segments = [
            ((d_width, 0), (roi_width - d_width, d_height)),
            ((0, 0), (d_width, roi_height // 2)),
            (
                (roi_width - int(0.9 * d_width), 0),
                (roi_width, roi_height // 2),
            ),
            (
                (d_width_center, roi_height // 2 - d_height_center),
                (
                    roi_width - d_width_center,
                    roi_height // 2 + d_height_center,
                ),
            ),
            ((0, roi_height // 2), (d_width, roi_height)),
            (
                (roi_width - int(0.9 * d_width), roi_height // 2),
                (roi_width, roi_height),
            ),
            ((0, roi_height - d_height), (roi_width - 10, roi_height)),
        ]
    elif point_side == "Right":
        segments = [
            ((0, 0), (roi_width - 30, d_height - 7)),
            ((0, 0), (d_width, roi_height // 2)),
            ((roi_width - d_width - 5, 0), (roi_width - 13, roi_height // 2)),
            (
                (0, roi_height // 2 - d_height_center - 7),
                (roi_width - 20, roi_height // 2 + d_height_center),
            ),
            ((0, roi_height // 2), (d_width, roi_height)),
            (
                (roi_width - d_width - 13, roi_height // 2),
                (roi_width - 13, roi_height),
            ),
            ((0, roi_height - d_height), (roi_width - 13, roi_height)),
        ]
    elif point_side == "Left":
        segments = [
            ((20, 0), (roi_width, d_height - 7)),
            ((20, 0), (d_width + 15, roi_height // 2)),
            ((roi_width - d_width + 5, 0), (roi_width, roi_height // 2)),
            (
                (20, roi_height // 2 - d_height_center - 7),
                (roi_width, roi_height // 2 + d_height_center),
            ),
            ((20, roi_height // 2), (d_width + 15, roi_height)),
            (
                (roi_width - d_width + 5, roi_height // 2),
                (roi_width, roi_height),
            ),
            ((20, roi_height - d_height), (roi_width, roi_height)),
        ]

    on_segments = [0] * len(segments)
    for (i, ((x_s_beginning, y_s_bottom), (x_s_final, y_s_top))) in enumerate(
        segments
    ):
        segment_roi = roi[y_s_bottom:y_s_top, x_s_beginning:x_s_final]
        pixels_total = cv2.countNonZero(segment_roi)
        segment_area = (x_s_final - x_s_beginning) * (y_s_top - y_s_bottom)
        if pixels_total / float(segment_area) > 0.45:
            on_segments[i] = 1
    return on_segments


def check_right_left_dot(roi):
    """To treat exception check left and right side for dot when contours join"""
    kernel = np.ones((2, 2), np.uint8)
    roi = cv2.dilate(roi, kernel, iterations=2)
    (roi_height, roi_width) = roi.shape
    d_width = int(roi_width * 0.24)

    segment_roi_left = roi[0:roi_height, 0 : d_width - 5]
    point_roi_left = segment_roi_left[roi_height - 30 : roi_height, :]
    segment_roi_right = roi[
        0:roi_height, roi_width - d_width + 15 : roi_width
    ]
    point_roi_right = segment_roi_right[roi_height - 30 : roi_height, :]
    if DEBUG:
        cv2.imshow("Left side", segment_roi_left)
        cv2.imshow("Left dot?", segment_roi_right)
        cv2.imshow("Right side", point_roi_left)
        cv2.imshow("Right dot?", point_roi_right)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    pixels_total_left = cv2.countNonZero(segment_roi_left)
    pixels_point_left = cv2.countNonZero(point_roi_left)
    pixels_total_right = cv2.countNonZero(segment_roi_right)
    pixels_point_right = cv2.countNonZero(point_roi_right)
    segment_area_left = (d_width - 5) * (roi_height)
    left_turn_on = pixels_total_left / float(segment_area_left)
    point_area_left = (d_width - 8) * (30)
    point_left_turn_on = pixels_point_left / float(point_area_left)
    segment_area_right = (d_width - 10) * (roi_height)
    right_turn_on = pixels_total_right / float(segment_area_right)
    point_area_right = (d_width - 15) * (30)
    point_right_turn_on = pixels_point_right / float(point_area_right)
    if left_turn_on < 0.30 and point_left_turn_on > 0.5:
        return "Left"
    if right_turn_on < 0.30 and point_right_turn_on > 0.5:
        return "Right"
    return "None"


def point_value_left(image, digit, x_beginning, y_bottom, d_width, height):
    """Check if the point is on the left side of the digit"""
    if digit in (3, 7):
        point_roi = image[
            y_bottom + height - height // 7 : y_bottom + height,
            x_beginning - 2 * d_width : x_beginning - d_width,
        ]
    else:
        point_roi = image[
            y_bottom + height - height // 7 : y_bottom + height,
            x_beginning - d_width + 2 : x_beginning - 2,
        ]
    if DEBUG:
        cv2.imshow("point roi", point_roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    area_point_roi = round((height // 7) * (d_width))
    pixels_point_roi = cv2.countNonZero(point_roi)

    if pixels_point_roi / float(area_point_roi) > 0.45:
        return True
    return False


def find_digits_value(digits_cnts, image):
    """Iterate over every contour and find its value"""
    digits_value = []

    for digit_cnt in digits_cnts:
        (x_beginning, y_bottom, width, height) = cv2.boundingRect(digit_cnt)
        if width < 50 and height > 80:
            x_final = x_beginning + width
            width = round(width * 3.9)
            x_beginning = x_final - width
            if x_beginning < 0:
                x_beginning = 0
                width = x_final

        roi = image[
            y_bottom : y_bottom + height, x_beginning : x_beginning + width
        ]
        kernel = np.ones((2, 2), np.uint8)
        roi = cv2.dilate(roi, kernel, iterations=1)

        if DEBUG:
            cv2.imshow("image", roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        point_side = "None"
        if height > 90:  # is a digit
            (roi_height, roi_width) = roi.shape
            (d_width, d_height) = (
                int(roi_width * 0.24),
                int(roi_height * 0.11),
            )

            on_segments = check_segments(roi)
            try:
                digit = DIGITS_LOOKUP[tuple(on_segments)]
            except:
                point_side = check_right_left_dot(roi)
                on_segments = check_segments(roi, point_side)
                try:
                    digit = DIGITS_LOOKUP[tuple(on_segments)]
                except:
                    print("ERRO")
                    exit()

            point_present = point_value_left(
                image, digit, x_beginning, y_bottom, d_width, height
            )
            if point_side == "Left" or point_present:
                digits_value.append(".")
                point_side = "None"
            digits_value.append(digit)
            if point_side == "Right":
                digits_value.append(".")
                point_side = "None"
        elif y_bottom < 90 and width > 30:
            digits_value.append("-")
        elif y_bottom > 90 and height < 60:
            digits_value.append(".")
    return digits_value


def save_value(img_res, digits_value_str):
    """Save image and result for later machine learning usage"""
    now = datetime.now()
    now_str = now.strftime("%d-%m-%Y_%Hh%Mmin%Ss")
    file_save = "Input/" + now_str + "_value=" + digits_value_str + ".png"
    cv2.imwrite(file_save, img_res)
    with open("Output/label.txt", "a") as label_file:
        label_file.write(file_save + " " + digits_value_str + "\n")

def ask_screen_points(img_res):
    """Ask the user the four points of where the screen is and do a 
       four point transform"""
    top_left = (float(input("X Top Left: ")), float(input("Y Top Left: ")))
    top_right = (float(input("X Top Right: ")), float(input("Y Top Right: ")))
    bottom_right = (float(input("X Bottom Right: ")), float(input("Y Bottom Right: ")))
    bottom_left = (float(input("X Bottom Left: ")), float(input("Y Bottom Left: ")))
    screen_box = np.array(
        [top_left, top_right, bottom_right, bottom_left],
        dtype="float32",
    )
    return four_point_transform(img_res, np.reshape(screen_box, (4, 2)))

DEBUG = str2bool(ARGS["debug"])
SCREENPOINT= str2bool(ARGS["screenPoints"])

def main():
    """Control program flow"""
    img_res = open_resize_image(RFACTOR)
    if not SCREENPOINT:
        screen_transform = find_screen(img_res)
    else:
        screen_transform = ask_screen_points(img_res)
    (y_image_shape, x_image_shape) = (
        screen_transform.shape[0],
        screen_transform.shape[1],
    )
    screen_digits = screen_transform[
        int(y_image_shape * YTB) : int(y_image_shape * YBT),
        int(x_image_shape * XLR) : int(x_image_shape * XRL),
    ]
    screen_digits_gray = cv2.cvtColor(
        screen_digits.copy(), cv2.COLOR_BGR2GRAY
    )
    digits_th = cv2.adaptiveThreshold(
        screen_digits_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        501,
        19,
    )
    digits_filtered = filter_screen(digits_th, 2, 3, 3, "closing")
    digits_cropp = cropp_to_digits(digits_filtered)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
    digits_cropp = cv2.erode(digits_cropp, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    digits_cropp = cv2.dilate(digits_cropp, kernel, iterations=1)
    digits_bitwise = cv2.bitwise_not(digits_cropp.copy())
    if DEBUG:
        cv2.imshow("Original Screen", open_resize_image(0.5))
        cv2.imshow("After fixed crop", screen_digits)
        cv2.imshow("Screen Multimeter", screen_transform)
        cv2.imshow("Digits", digits_bitwise)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    digits_cnts = find_digits_cnts(digits_bitwise)
    digits_value = find_digits_value(digits_cnts, digits_bitwise)
    digits_value_str = "".join(map(str, digits_value))
    print(digits_value_str)
    save_value(img_res, digits_value_str)


if __name__ == "__main__":
    main()
