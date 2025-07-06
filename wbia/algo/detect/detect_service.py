import logging
import os
import requests

logger = logging.getLogger('wbia')


def run_inference_on_image(img_gid, image_path, model_tag):

    service_name = os.getenv("detector_service_name", "detector_inference")
    service_port = os.getenv("detector_service_PORT", "6050")

    url = f"http://{service_name}:{service_port}/predict"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    data = {
        "model_id": model_tag,
        "image_uri": image_path,
    }

    logger.info("Sending POST request for path %s to URL: %s", image_path, url)
    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        logger.warning("Failed inference for path %s: %s", image_path, response.text)
        return [], [], [], [], [], []

    response_json = response.json()
    bbox_list = [[round(x) for x in bbox] for bbox in response_json['bboxes']]
    theta_list = response_json['thetas']
    class_list = response_json['class_names']
    conf_list = response_json['scores']
    notes_list = ['service'] * len(bbox_list)
    gid_list = [img_gid] * len(bbox_list)
    return gid_list, bbox_list, theta_list, class_list, conf_list, notes_list
