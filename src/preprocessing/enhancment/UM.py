from Tools import *

class UM:

    def __init__(
        self,
        filter=1,
        radius=5,
        amount=2,
        ):
        self.filter = filter
        self.radius = radius
        self.amount = amount

    def set_filter(self, filter):
        self.filter = filter

    def set_radius(self, radius):
        self.radius = radius

    def set_amount(self, amount):
        self.amount = amount

    def run(self, image):
        image = img_as_float(image)  # ensuring float values for computations
        if self.filter == 1:
            blurred_image = gaussian_filter(image, sigma=self.radius)
        elif self.filter == 2:
            blurred_image = median_filter(image, size=20)
        elif self.filter == 3:
            blurred_image = maximum_filter(image, size=20)
        else:
            blurred_image = minimum_filter(image, size=20)

        mask = image - blurred_image  # keep the edges created by the filter
        sharpened_image = image + mask * self.amount

        sharpened_image = np.clip(sharpened_image, float(0), float(1))  # Interval [0.0, 1.0]
        sharpened_image = (sharpened_image * 255).astype(np.uint8)  # Interval [0,255]

        return sharpened_image