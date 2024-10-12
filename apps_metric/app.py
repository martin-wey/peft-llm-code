import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("loubnabnl/apps_metric")
launch_gradio_widget(module)