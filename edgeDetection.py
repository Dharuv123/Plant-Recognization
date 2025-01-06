import numpy as np
import cv2

class EdgeDetection:
    @staticmethod
    def auto_canny(image):
        med = np.median(image)
        lower = int(max(0, (1.0 - 0.33) * med))
        upper = int(min(255, (1.0 + 0.33) * med))
        return cv2.Canny(image, lower, upper)

    def show_image(self, image_path):
        orig_image = cv2.imread(image_path)
        
        # Check if the image was loaded correctly
        if orig_image is None:
            print("Error: Could not load image. Check the file path.")
            return

        gray_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
        wide_image = cv2.Canny(blurred, 10, 200)
        tight_image = cv2.Canny(blurred, 225, 250)
        auto_image = self.auto_canny(blurred)

        cv2.imshow("Original", orig_image)
        cv2.imshow("Edge Detection", np.hstack([wide_image, tight_image, auto_image]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = input("Enter the image path: ")
    edge_detector = EdgeDetection()
    edge_detector.show_image(image_path)
