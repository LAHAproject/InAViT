"""Tools for visualising hand-object detections"""
import os
import warnings
from copy import deepcopy
from typing import Tuple

import PIL.Image
from PIL import ImageFont, ImageDraw

"""The core set of types that represent hand-object detections"""

from enum import Enum, unique
from itertools import chain
from typing import Dict, Iterator, List, Tuple, cast

import numpy as np


class BBox():
    """
    left: float
    top: float
    right: float
    bottom: float
    """
    def __init__(self, box):
        self.left = box[0]
        self.top = box[1]
        self.right = box[2]
        self.bottom = box[3]
        
    @property
    def center(self) -> Tuple[float, float]:
        x = (self.left + self.right) / 2
        y = (self.top + self.bottom) / 2
        return x, y

    @property
    def center_int(self) -> Tuple[int, int]:
        """Get center position as a tuple of integers (rounded)"""
        x, y = self.center
        return (round(x), round(y))

    def scale(self, width_factor: float = 1, height_factor: float = 1) -> None:
        self.left *= width_factor
        self.right *= width_factor
        self.top *= height_factor
        self.bottom *= height_factor

    def center_scale(self, width_factor: float = 1, height_factor: float = 1) -> None:
        x, y = self.center
        new_width = self.width * width_factor
        new_height = self.height * height_factor
        self.left = x - new_width / 2
        self.right = x + new_width / 2
        self.top = y - new_height / 2
        self.bottom = y + new_height / 2

    @property
    def coords(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (
            self.top_left,
            self.bottom_right,
        )

    @property
    def coords_int(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (
            self.top_left_int,
            self.bottom_right_int,
        )

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def top_left(self) -> Tuple[float, float]:
        return (self.left, self.top)

    @property
    def bottom_right(self) -> Tuple[float, float]:
        return (self.right, self.bottom)

    @property
    def top_left_int(self) -> Tuple[int, int]:
        return (round(self.left), round(self.top))

    @property
    def bottom_right_int(self) -> Tuple[int, int]:
        return (round(self.right), round(self.bottom))


class ObjectDetection():
    """Dataclass representing an object detection, consisting of a bounding box and a
    score (the model's confidence this is an object)"""
    """
    bbox: BBox
    score: np.float32
    """
    def __init__(self, box) -> None:
        self.bbox = BBox(box[:4])
        self.score = box[4]

    def scale(self, width_factor: float = 1, height_factor: float = 1) -> None:
        self.bbox.scale(width_factor=width_factor, height_factor=height_factor)

    def center_scale(self, width_factor: float = 1, height_factor: float = 1) -> None:
        self.bbox.center_scale(width_factor=width_factor, height_factor=height_factor)



class FrameDetections():
    """Dataclass representing hand-object detections for a frame of a video"""
    """
    video_id: str
    frame_number: int
    objects: List[ObjectDetection]
    hands: List[ObjectDetection]
    """
    def __init__(self, boxes, video_id, frame_number) -> None:
        self.video_id = video_id
        self.frame_number = frame_number
        self.objects = []
        self.hands = []

        if boxes['obj'] is not None:
            self.objects = [ObjectDetection(boxobj) for boxobj in boxes['obj']]
        if boxes['hand'] is not None:
            self.hands = [ObjectDetection(boxhand) for boxhand in boxes['hand']]
        
    def scale(self, width_factor: float = 1, height_factor: float = 1) -> None:
        """
        Scale the coordinates of all the hands/objects. x components are multiplied
        by the ``width_factor`` and y components by the ``height_factor``
        """
        for det in chain(self.hands, self.objects):
            det.scale(width_factor=width_factor, height_factor=height_factor)

    def center_scale(self, width_factor: float = 1, height_factor: float = 1) -> None:
        """
        Scale all the hands/objects about their center points.
        """
        for det in chain(self.hands, self.objects):
            det.center_scale(width_factor=width_factor, height_factor=height_factor)

def load_detections(path):
    import pickle
    pkl = pickle.load(open(path, "rb"))
    vid = path.split('/')[1].strip('.pkl')
    
    return [FrameDetections(pkl[frame], vid, frame) for frame in pkl]

class DetectionRenderer:
    """A class to render hand-object annotations onto the corresponding image"""
    def __init__(
        self,
        hand_threshold: float = 0.01,
        object_threshold: float = 0.01,
        font_size=20,
        border=4,
        text_padding=4,
    ):
        """

        Args:
            hand_threshold: Filter hand detections above this threshold
            object_threshold: Filter object detections above this threshold
            only_interacted_objects: Only draw objects that are part of an
                interaction with a hand
            font_size: The font-size for the bounding box labels
            border: The width of the border of annotation bounding boxes.
            text_padding: The amount of padding within bounding box annotation labels
        """
        self.hand_threshold = hand_threshold
        self.object_threshold = object_threshold
        
        try:
            self.font = ImageFont.truetype(
                os.path.dirname("." + "/Roboto-Regular.ttf"),
                size=font_size,
            )
        except IOError:
            warnings.warn(
                "Could not find font, falling back to Pillow default. "
                "`font_size` will not have an effect"
            )
            self.font = ImageFont.load_default()
        self.hand_rgb = (0, 90, 181)
        self.border = border
        self.text_padding = text_padding
        
        self.object_rgb = (255, 194, 10)
        self.object_rgba = (*self.object_rgb, 70)
        self._img: PIL.Image.Image
        self._detections: FrameDetections
        self._draw: ImageDraw.ImageDraw

        
    def render_detections(
        self, frame: PIL.Image.Image, detections: FrameDetections
    ) -> PIL.Image.Image:
        """
        Args:
            frame: Frame to annotate with hand and object detections
            detections: Detections for the current frame

        Returns:
            A copy of ``frame`` annotated with the detections from ``detections``.
        """
        self._img = frame.copy()
        detections = self._detections = deepcopy(detections)
        # detections.scale(
        #     width_factor=self._img.width, height_factor=self._img.height
        # )
        
        if len(detections.hands) == 0 and len(detections.objects) == 0:
            return self._img

        self._draw = ImageDraw.Draw(frame)
        
        if len(detections.objects) > 0:
            print(detections.objects[0].bbox.top_left)
            for object in detections.objects:
                if object.score >= self.object_threshold:
                    self._render_object(object)
        
        if len(detections.hands) > 0:
            print(detections.hands[0].bbox.top_left)
            for hand in detections.hands:
                if hand.score >= self.hand_threshold:
                    self._render_hand(hand)

        return self._img

    def _render_hand(self, hand: ObjectDetection):
        mask = PIL.Image.new("RGBA", self._img.size)
        mask_draw = ImageDraw.Draw(mask)
        hand_bbox = hand.bbox.coords_int
        color = self.hand_rgb
        mask_draw.rectangle(
            hand_bbox,
            outline=color,
            width=self.border,
            #fill=self.hand_rgb,
            fill=None
        )
        self._img.paste(mask, (0, 0), mask)
        self._render_label_box(
            ImageDraw.Draw(self._img),
            top_left=hand.bbox.top_left_int,
            text="H",
            padding=self.text_padding,
            outline_color=color,
        )

    def _render_object(self, object: ObjectDetection):
        mask = PIL.Image.new("RGBA", self._img.size)
        mask_draw = ImageDraw.Draw(mask)
        object_bbox = object.bbox.coords_int
        mask_draw.rectangle(
            object_bbox,
            outline=self.object_rgb,
            width=self.border,
            #fill=self.object_rgba,
            fill=None
        )
        self._img.paste(mask, (0, 0), mask)
        self._render_label_box(
            ImageDraw.Draw(self._img),
            top_left=object.bbox.top_left_int,
            text="O",
            padding=self.text_padding,
            outline_color=self.object_rgb,
        )
        
    def _render_label_box(
        self,
        draw: ImageDraw.ImageDraw,
        top_left: Tuple[int, int],
        text: str,
        padding: int = 10,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        outline_color: Tuple[int, int, int] = (0, 0, 0),
        text_color: Tuple[int, int, int] = (0, 0, 0),
    ):
        text_size = draw.textsize(text, font=self.font)
        #offset_x, offset_y = self.font.getlength(text)
        text_width = text_size[0] 
        text_height = text_size[1] 
        x, y = top_left
        bottom_right = (
            x + self.border * 2 + padding * 2 + text_width,
            y + padding + text_height,
        )
        box_coords = [top_left, bottom_right]
        draw.rectangle(
            box_coords, 
            #fill=background_color,
            outline=outline_color,
            width=self.border,
        )
        text_coordinate = (
            x + self.border + padding,
            y + self.border + padding + 1,
        )
        draw.text(text_coordinate, text, font=self.font, fill=text_color)