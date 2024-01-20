import importlib
import cv2
import argparse


parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
parser.add_argument('--tracker_name', type=str, help='Name of tracking method.')
parser.add_argument('--tracker_param', type=str, help='Name of parameter file.')
parser.add_argument('--input_video', type=str, help='Path to input video.')
parser.add_argument('--output_video', type=str, help='Path to output video.')
parser.add_argument('--init_bbox', nargs="*", type=int, help='Initial target bounding box')
parser.add_argument('--language', type=str, help='Language description of target')
args = parser.parse_args()


def _read_image(image_file):
    if isinstance(image_file, str):
        im = cv2.imread(image_file)
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

input_video = args.input_video
output_video = args.output_video
parameter_name = args.tracker_param

init_info = {}
# specify target reference
init_info['language'] = args.language # for NL and NLBBOX mode
init_info['init_bbox'] = args.init_bbox # for BBOX and NLBBOX mode

param_module = importlib.import_module(f'lib.test.parameter.{args.tracker_name}')
params = param_module.parameters(parameter_name, None)
params.debug = False

tracker_class = importlib.import_module(f'lib.test.tracker.{args.tracker_name}').get_tracker_class()
tracker = tracker_class(params, '')

output = {'target_bbox': [],
            'time': []}
if tracker.params.save_all_boxes:
    output['all_boxes'] = []
    output['all_scores'] = []

def _store_outputs(tracker_out: dict, defaults=None):
    defaults = {} if defaults is None else defaults
    for key in output.keys():
        val = tracker_out.get(key, defaults.get(key, None))
        if key in tracker_out or val is not None:
            output[key].append(val)

def _store_outputs(tracker_out: dict, defaults=None):
    defaults = {} if defaults is None else defaults
    for key in output.keys():
        val = tracker_out.get(key, defaults.get(key, None))
        if key in tracker_out or val is not None:
            output[key].append(val)

videoCapture = cv2.VideoCapture(input_video)
success, image = videoCapture.read()

out = tracker.initialize(image, init_info)

height, weight, _ = image.shape
fps = 20
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videowriter = cv2.VideoWriter(output_video, fourcc, fps, (weight, height))
success, image = videoCapture.read()
while success:
    info = {}
    out = tracker.track(image, info)
    x, y, w, h = out['target_bbox']
    image = cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0))
    videowriter.write(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    success, image = videoCapture.read()
videowriter.release()