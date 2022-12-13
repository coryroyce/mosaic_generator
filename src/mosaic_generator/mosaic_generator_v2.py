from pathlib import Path
import json
import math
import random

import numpy as np
import cv2

# Set configuration parameters
SOURCE_IMAGE_PATH = "src/mosaic_generator/source_images/B.jpg"  # "src/mosaic_generator/source_images/Pikachu.jpeg"
TILE_IMAGES_PATH = "src/mosaic_generator/tile_images"
TILE_IMAGES_CACHE_PATH = "src/mosaic_generator/tile_images_cache/tile_images_cache.json"
OUTPUT_IMAGE_PATH = "src/mosaic_generator/output_mosaics"
REGENERATE_NEW_CACHE = False
NUMBER_OF_TILES_HIGH = 21
NUMBER_OF_TILES_WIDE = 16


class ProcessImage:
    def __init__(self):
        self.source_image_path = SOURCE_IMAGE_PATH
        self.tile_images_path = TILE_IMAGES_PATH
        self.tile_images_cache_path = TILE_IMAGES_CACHE_PATH
        self.output_image_path = self.create_output_file_name()
        self.regenerate_new_cache = REGENERATE_NEW_CACHE
        self.number_of_tiles_high = NUMBER_OF_TILES_HIGH
        self.number_of_tiles_wide = NUMBER_OF_TILES_WIDE
        self.source_image_original = None
        self.source_image_modified = None
        self.source_image_height = 0
        self.source_image_width = 0
        self.tile_images_data = {}
        self.tile_height = 47
        self.tile_width = 63
        self.tiles = None

    def process(self):
        """Main process function that combines all the needed functions in order
        to generate the custom photo mosaic"""
        # Process the tile images and cache data if needed
        self.check_cache_for_tile_images()

        # Load and modify the source image
        self.load_and_modify_source_image()

        # Generate tiles
        self.generate_tiles()

        # Generate the Mosaic
        self.generate_mosaic()

        return

    @staticmethod
    def get_average_color(image):
        """Pass in any image and get the average color in terms of RGB for that image

        Returns:
            average_color: tuple Ex: (10, 50. 255)
        """
        average_color = np.average(np.average(image, axis=0), axis=0)
        average_color = np.around(average_color, decimals=-1)
        average_color = tuple(int(i) for i in average_color)
        return average_color

    @staticmethod
    def get_closest_color(color, colors):
        cr, cg, cb = color

        min_difference = float("inf")
        closest_color = None
        for c in colors:
            r, g, b = eval(c)
            difference = math.sqrt((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2)
            if difference < min_difference:
                min_difference = difference
                closest_color = eval(c)

        return closest_color

    def create_output_file_name(self):
        "Create the output file name based on the image source"
        input_file_name = Path(self.source_image_path).stem
        output_file_name = f"{OUTPUT_IMAGE_PATH}/{input_file_name}_Mosaic.jpeg"

        return output_file_name

    def check_cache_for_tile_images(self):
        "Check if a cache exists or if should be regenerated based on new tile images"
        if not Path(self.tile_images_cache_path).is_file() or self.regenerate_new_cache:
            # Create a list of all picture file types
            image_paths = Path(self.tile_images_path).glob("*")
            images = []
            for image_path in image_paths:
                if image_path.suffix.lower() in [".jpeg", ".jpg", ".png"]:
                    # print(image.suffix)
                    images.append(str(image_path))

            # Generate the image average color data to be cached
            data = {}
            for img_path in images:
                img = cv2.imread(str(img_path))
                average_color = self.get_average_color(img)
                if str(tuple(average_color)) in data:
                    data[str(tuple(average_color))].append(str(img_path))
                else:
                    data[str(tuple(average_color))] = [str(img_path)]

            # Create Cache file
            with open(self.tile_images_cache_path, "w") as file:
                json.dump(data, file, indent=2, sort_keys=True)
            print(f"Caching done")

        # Load in cached data to the tile images data
        with open(self.tile_images_cache_path, "r") as file:
            tile_images_data = json.load(file)

        self.tile_images_data = tile_images_data

        return tile_images_data

    def find_nearest_aspect_ratio(self, image):
        """Take an input image and find the nearest aspect ratio from 9:16, 16:9 or 1:1"""
        image_original = cv2.imread(self.source_image_path)

        # Aspect ratio is width:height
        input_aspect_ratio = image_original.shape[1] / image_original.shape[0]

        # 9:16 = 0.5625 AR, 1:1 = 1 AR, 16:9 = 1.7777
        # Midpoints 0.7813 and 1.3888
        if input_aspect_ratio <= 0.7813:
            aspect_ratio = "9:16"
        elif input_aspect_ratio > 0.7813 and input_aspect_ratio <= 1.3888:
            aspect_ratio = "1:1"
        elif input_aspect_ratio >= 1.3888:
            aspect_ratio = "16:9"

        return aspect_ratio

    @staticmethod
    def center_crop_image(image):
        """Returns center cropped image
        Args:
        img: image to be center cropped
        dim: dimensions (width, height) to be cropped
        """
        width, height = image.shape[1], image.shape[0]

        # Default to square dimensions can add other aspect ratios later
        dim = [min(width, height), min(width, height)]

        # Process crop width and height for max available dimension
        crop_width = dim[0] if dim[0] < image.shape[1] else image.shape[1]
        crop_height = dim[1] if dim[1] < image.shape[0] else image.shape[0]
        mid_x, mid_y = int(width / 2), int(height / 2)
        cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
        cropped_image = image[mid_y - ch2 : mid_y + ch2, mid_x - cw2 : mid_x + cw2]
        return cropped_image

    def load_and_modify_source_image(self):
        """Load in the source image and then modify based on overall parameters"""
        image_original = cv2.imread(self.source_image_path)
        self.source_image_original = image_original

        # TODO: Calculate resize factor for just the width
        # Standard aspect ratio's = 9:16, 16:9 or 1:1

        # Crop the image based on the best aspect ratio
        image_modified = self.center_crop_image(image_original)

        # Calculate output height and width of image
        output_image_height = self.number_of_tiles_high * self.tile_height
        output_image_width = self.number_of_tiles_wide * self.tile_width

        # Resize the image
        output_dimentions = (output_image_width, output_image_height)
        image_modified = cv2.resize(
            image_modified, output_dimentions, interpolation=cv2.INTER_NEAREST
        )

        # Update source image height and width
        self.source_image_height, self.source_image_width, _ = image_modified.shape

        self.source_image_modified = image_modified

        return image_modified

    def generate_tiles(self):
        """Generate the tiles used to update the input image"""
        tiles = []
        for y in range(0, self.source_image_height, self.tile_height):
            for x in range(0, self.source_image_width, self.tile_width):
                tiles.append((y, y + self.tile_height, x, x + self.tile_width))

        self.tiles = tiles

        return tiles

    def generate_mosaic(self):
        """Generate the mosaic of the source file using each tile"""
        for tile in self.tiles:
            y0, y1, x0, x1 = tile

            # Get the average tile color for the source image
            try:
                average_color = self.get_average_color(
                    self.source_image_modified[y0:y1, x0:x1]
                )
            except Exception:
                continue

            # Find the closest matching tile image to that average color
            closest_color = self.get_closest_color(
                average_color, self.tile_images_data.keys()
            )

            tile_image_path = random.choice(self.tile_images_data[str(closest_color)])
            tile_image = cv2.imread(tile_image_path)
            tile_image = cv2.resize(tile_image, (self.tile_width, self.tile_height))
            self.source_image_modified[y0:y1, x0:x1] = tile_image

            # # Uncomment to sho image being generated tile by tile
            # cv2.imshow("Image", self.source_image_modified)
            # cv2.waitKey(1)

        cv2.imwrite(self.output_image_path, self.source_image_modified)


def test_01():
    # # Check path or files existence
    # path_to_check = Path(TILE_IMAGES_CACHE_PATH)
    # print(path_to_check.is_file())

    # # Test image reading
    # img = cv2.imread(SOURCE_IMAGE_PATH)
    # cv2.imshow("Image", img)
    # cv2.waitKey()

    # Process the tile images or test single function
    process = ProcessImage()
    # process.check_cache_for_tile_images()
    process.process()
    # result = process.get_average_color(img)
    # print(result)

    # # List all files from tile images
    # images = Path(TILE_IMAGES_PATH).glob("*")
    # images_list = []
    # for image in images:
    #     if image.suffix.lower() in [".jpeg", ".jpg", ".png"]:
    #         # print(image.suffix)
    #         images_list.append(str(image))
    # print(images_list)

    return


if __name__ == "__main__":
    test_01()
    # if len(sys.argv) < 3:
    #     show_error("Usage: {} <image> <tiles directory>\r".format(sys.argv[0]))
    # else:
    #     source_image = sys.argv[1]
    #     tile_dir = sys.argv[2]
    #     if not os.path.isfile(source_image):
    #         show_error("Unable to find image file '{}'".format(source_image))
    #     elif not os.path.isdir(tile_dir):
    #         show_error("Unable to find tile directory '{}'".format(tile_dir))
    #     else:
    #         mosaic(source_image, tile_dir)
