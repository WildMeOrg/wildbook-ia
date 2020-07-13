# -*- coding: utf-8 -*-
"""Dependencies: flask, tornado."""
from __future__ import absolute_import, division, print_function
from wbia.control import controller_inject
from flask_swagger import swagger
import wbia.constants as const
from flask import current_app
from flask import jsonify
from flask import url_for
import utool as ut
import traceback
import uuid

(print, rrr, profile) = ut.inject2(__name__)

CLASS_INJECT_KEY, register_ibs_method = controller_inject.make_ibs_register_decorator(
    __name__
)

PREFIX = controller_inject.MICROSOFT_API_PREFIX
register_api = controller_inject.get_wbia_flask_api(__name__)
register_route = controller_inject.get_wbia_flask_route(__name__)


def _prefix(route=''):
    rule = '/%s/%s/' % (PREFIX, route,)
    while '//' in rule:
        rule = rule.replace('//', '/')
    return rule


def _image(ibs, gid):
    return {
        'uuid': str(ibs.get_image_uuids(gid)),
    }


def _annotation(ibs, aid):
    return {
        'uuid': str(ibs.get_annot_uuids(aid)),
    }


def _name(ibs, nid):
    return {
        'uuid': str(ibs.get_name_uuids(nid)),
    }


# DEPRICATE
#     def _detection(ibs, aid):
#         gid = ibs.get_annot_gids(aid)
#         bbox = ibs.get_annot_bboxes(aid)
#         return {
#             '_image'      : _image(ibs, gid),
#             '_annotation' : _annotation(ibs, aid),
#             'xtl'         : bbox[0],
#             'ytl'         : bbox[1],
#             'width'       : bbox[2],
#             'height'      : bbox[3],
#             'theta'       : ibs.get_annot_thetas(aid),
#             'label'       : ibs.get_annot_species_texts(aid),
#             'score'       : ibs.get_annot_detect_confidence(aid),
#         }


def _detection(ibs, gid, result):
    bbox, theta, conf, label = result
    detection = {
        'image': _image(ibs, gid),
        'bbox': bbox,
        'xtl': bbox[0],
        'ytl': bbox[1],
        'xbr': bbox[0] + bbox[2],
        'ybr': bbox[1] + bbox[3],
        'width': bbox[2],
        'height': bbox[3],
        'theta': theta,
        'species': label,
        'viewpoint': None,
        'score': conf,
    }
    return detection


def _task(ibs, taskid):
    return {
        'uuid': taskid,
    }


def _ensure_general(ibs, models, tag, rowid_from_uuid_func, unpack=True, *args, **kwargs):

    if isinstance(models, dict):
        models = [models]

    length = len(models)
    single = length == 1
    if length == 0:
        return []

    rowid_list = []
    for index, model in enumerate(models):
        try:
            if single:
                parameter = 'models'
            else:
                parameter = 'models:%d' % (index,)

            assert 'uuid' in model, '%s Model provided is invalid, missing UUID key' % (
                tag,
            )

            uuid_ = uuid.UUID(model['uuid'])
            assert uuid_ is not None, "%s Model's UUID is invalid" % (tag,)

            rowid = rowid_from_uuid_func(uuid_)
            assert rowid is not None, '%s Model is unrecognized, please upload' % (tag,)
        except AssertionError as ex:
            raise controller_inject.WebInvalidInput(str(ex), parameter)
        rowid_list.append(rowid)

    if single and unpack:
        return rowid_list[0]
    else:
        return rowid_list


def _ensure_images(ibs, images, *args, **kwargs):
    return _ensure_general(
        ibs, images, 'Image', ibs.get_image_gids_from_uuid, *args, **kwargs
    )


def _ensure_annotations(ibs, annotations, *args, **kwargs):
    return _ensure_general(
        ibs, annotations, 'Annotation', ibs.get_annot_aids_from_uuid, *args, **kwargs
    )


def _ensure_names(ibs, names, *args, **kwargs):
    return _ensure_general(
        ibs, names, 'Name', ibs.get_name_rowids_from_uuid, *args, **kwargs
    )


@register_route(_prefix('swagger'), methods=['GET'])
def microsoft_core_specification_swagger(*args, **kwargs):
    r"""
    Returns the API specification in the Swagger 2.0 (OpenAPI) JSON format.

    The Swagger API specification (https://github.com/OAI/OpenAPI-Specification/blob/master/versions/2.0.md) provides a standardized method to export REST API documentation and examples.  Our documentation is built on-demand with the help of the Python package flask-swagger (https://github.com/gangverk/flask-swagger).

    The API specification includes GET, POST, PUT, and DELETE methods and Model definitions.
    ---
    definitions:
    - schema:
        id: Image
        description: An Image is a semantic construct that represents an uploaded image.  Images can be uploaded for later processing or be used immediately for detection.  Object detection will create Annotation models, which have a required Image parent.  An Image can have multiple detections (i.e. Annotation models).
        required:
          - uuid
        properties:
          uuid:
            description: a deterministically-derived UUID based on the image pixels, which can be used to identify duplicate Images.
            type: string
            format: uuid
    - schema:
        id: Annotation
        description: An Annotation is a semantic construct that represents a *committed* detection, with a bounding box and species classification assigned as stored attributes.  An Annotation is required to have a parent Image.  All bounding box coordinates are relative to the size of the parent Image that was uploaded.
        required:
          - uuid
          - bbox
        properties:
          uuid:
            description: a deterministically-derived UUID based on the parent image's UUID and the bounding box coordinate (xtl, ytl, width, height) and orientation (theta), which can be used to identify duplicate Annotations.
            type: string
            format: uuid
          bbox:
            description: a 4-tuple of coordinates that defines a rectangle in the format (x-axis top left corner, y-axis top left corner, width, height) in pixels.  These values are expected to be bounded by the size of the parent image.
            type: array
          theta:
            description: a rotation around the center of the annotation, in radians
            type: number
          species:
            description: a user-defined string to specify the species of the annotation (e.g. 'zebra' or 'massai_giraffe').  This value is used to filter matches and run-time models for ID.
            type: string
          viewpoint:
            description: a user-defined string to specify the viewpoint of the annotation (e.g. 'right' or 'front_left').  This value is used to filter matches and run-time models for ID.
            type: string
          name:
            description: the name of the individual
            format: uuid
            type: string
    - schema:
        id: Detection
        description: A Detection is a semantic constrict that represents an *un-committed* detection.  A Detection can be committed to an Annotation to be stored permanently on the parent Image.
        required:
          - _image
          - score
          - bbox
          - xtl
          - ytl
          - xbr
          - ybr
          - width
          - height
          - theta
          - label
        properties:
          image:
            description: The Image that this Detection was found in
            $ref: "#/definitions/Image"
          score:
            description: The detection's classification score
            type: integer
            format: int32
          bbox:
            description: The bounding box for this annotation, represented in the format (xtl, ytl, width, height)
            type: array
            items:
              type: number
              format: float
          xtl:
            description: The pixel coordinate for the top-left corner along the x-axis (xtl = x-axis top left) for the bounding box
            type: integer
            format: int32
          ytl:
            description: The pixel coordinate for the top-left corner along the y-axis (ytl = y-axis top left) for the bounding box
            type: integer
            format: int32
          xbr:
            description: The pixel coordinate for the bottom-right corner along the x-axis (ytl = x-axis bottom right) for the bounding box
            type: integer
            format: int32
          ybr:
            description: The pixel coordinate for the bottom-right corner along the y-axis (ytl = y-axis bottom right) for the bounding box
            type: integer
            format: int32
          width:
            description: The pixel width for the bounding box
            type: integer
            format: int32
          height:
            description: The pixel height for the bounding box
            type: integer
            format: int32
          theta:
            description: The rotation of the bounding box around its center, represented in radians
            type: number
            format: float
          species:
            description: The semantic species classification (class label) of the bounding box
            type: string
          viewpoint:
            description: The semantic viewpoint classification (class label) of the bounding box
            type: string
    - schema:
        id: Name
        description: A Name is the identification label for a group of Annotations
        required:
          - uuid
          - alias
        properties:
          uuid:
            description: a deterministically-derived UUID based on the image pixels, which can be used to identify duplicate Images.
            type: string
            format: uuid
          alias:
            description: a string alias for this individual, helpful for user-facing interfaces
            type: string
    - schema:
        id: Task
        description: A Task is a semantic construct that represents a background task (i.e. detection) in an asynchronous call.  A Task has an optional callback on completion or the status (and result) can be checked via the API
        required:
          - uuid
        properties:
          uuid:
            description: a random UUID to identify a given asynchronous call, used to check status and results of a background task
            type: string
            format: uuid
    produces:
    - application/json
    responses:
        200:
          description: Returns the Swagger 2.0 JSON format
    """
    try:
        swag = swagger(current_app)
    except Exception:
        print(str(traceback.format_exc()))
        # ut.embed()

    swag['info']['title'] = 'Wild Me - IA (Image Analysis)'
    swag['info'][
        'description'
    ] = 'Documentation for all classification, detection, and identification calls provided by Wild Me for the AI for Earth (AI4E) collaboration'
    swag['info']['version'] = 'v0.1'
    swag['info']['contact'] = {
        'name': 'Wild Me Developers (AI4E)',
        'url': 'http://wildme.org',
        'email': 'dev@wildme.org',
    }
    swag['info']['license'] = {
        'name': 'Apache 2.0',
        'url': 'http://www.apache.org/licenses/LICENSE-2.0.html',
    }
    swag['host'] = 'demo.wildbook.org:5010'
    swag['schemes'] = [
        'http',
    ]
    # swag['basePath'] = PREFIX

    # "securityDefinitions": {
    #   "apiKeyHeader": {
    #     "type": "apiKey",
    #     "name": "Ocp-Apim-Subscription-Key",
    #     "in": "header"
    #   },
    #   "apiKeyQuery": {
    #     "type": "apiKey",
    #     "name": "subscription-key",
    #     "in": "query"
    #   }
    # },
    # "security": [
    #   {
    #     "apiKeyHeader": []
    #   },
    #   {
    #     "apiKeyQuery": []
    #   }
    # ],

    response = jsonify(swag)
    return response


@register_api(_prefix('status'), methods=['GET'], __api_plural_check__=False)
def microsoft_core_status(ibs, *args, **kwargs):
    r"""
    Returns the health status of the API back-end; optionally can be used as a service heatbeat.
    ---
    produces:
    - application/json
    responses:
      200:
        description: Returns the status of the server
        schema:
          type: object
          properties:
            status:
              type: string
              enum:
              - healthy
              - warning
              - critical
        examples:
          application/json:
            status: healthy
    """
    ibs.heartbeat()
    status = 'healthy'
    return {'status': status}


@register_api(_prefix('image'), methods=['POST'])
def microsoft_image_upload(ibs, *args, **kwargs):
    r"""
    Upload an Image to the system and return the UUID
    ---
    parameters:
    - name: image
      in: formData
      description: The image to upload.
      required: true
      type: file
      enum:
      - image/png
      - image/jpg
      - image/tiff
    consumes:
    - multipart/form-data
    produces:
    - application/json
    responses:
      200:
        description: Returns an Image model with a UUID
        schema:
          $ref: "#/definitions/Image"
      400:
        description: Invalid input parameter
      415:
        description: Unsupported media type in the request body. Currently only image/png, image/jpeg, image/tiff are supported.
    """
    from wbia.web.apis import image_upload

    try:
        gid = image_upload(cleanup=True, **kwargs)
        assert gid is not None
    except controller_inject.WebException:
        raise
    except Exception:
        raise controller_inject.WebInvalidInput(
            'Uploaded image is corrupted or is an unsupported file format (supported: image/png, image/jpeg, image/tiff)',
            'image',
            image=True,
        )
    return _image(ibs, gid)


@register_api(_prefix('annotation'), methods=['POST'])
def microsoft_annotation_add(
    ibs,
    image,
    bbox,
    theta=None,
    species=None,
    viewpoint=None,
    name=None,
    *args,
    **kwargs,
):
    r"""
    Add an Annotation to the system and return the UUID
    ---
    parameters:
    - name: image
      in: body
      description: A Image model for which the annotation will be added as a child
      required: true
      schema:
        $ref: "#/definitions/Image"
    - name: bbox
      in: formData
      description: a 4-tuple of coordinates that defines a rectangle in the format (x-axis top left corner, y-axis top left corner, width, height) in pixels.  These values are expected to be bounded by the size of the parent image.
      required: true
      type: array
      items:
        type: number
        format: float
    - name: theta
      in: formData
      description: a rotation around the center of the annotation, in radians.  Default is 0.0
      required: false
      type: number
    - name: species
      in: formData
      description: a user-defined string to specify the species of the annotation (e.g. 'zebra' or 'massai_giraffe').  This value is used to filter matches and run-time models for ID.  Default is None.
      required: false
      type: string
    - name: viewpoint
      in: formData
      description: a user-defined string to specify the viewpoint of the annotation (e.g. 'right' or 'front_left').  This value is used to filter matches and run-time models for ID. Default is None.
      required: false
      type: string
    - name: name
      in: body
      description: a Name model to assign to the created Annotation
      required: false
      schema:
        $ref: "#/definitions/Name"
    consumes:
    - multipart/form-data
    produces:
    - application/json
    responses:
      200:
        description: Returns an Annotations model with a UUID
        schema:
          $ref: "#/definitions/Annotation"
      400:
        description: Invalid input parameter
    """
    try:
        parameter = 'bbox'
        assert isinstance(bbox, (tuple, list)), 'Bounding box must be a list'
        assert len(bbox) == 4, 'Bounding Box must have 4 yntegers'
        assert isinstance(bbox[0], int), 'Bounding box xtl (index 0) must be an integer'
        assert isinstance(bbox[1], int), 'Bounding box ytl (index 1) must be an integer'
        assert isinstance(bbox[2], int), 'Bounding box width (index 2) must be an integer'
        assert isinstance(
            bbox[3], int
        ), 'Bounding box height (index 3) must be an integer'

        if theta is not None:
            assert isinstance(theta, float), 'Theta must be a float'

        if species is not None:
            assert isinstance(species, str), 'Species must be a string'
            assert len(species) > 0, 'Species cannot be empty'

        if viewpoint is not None:
            assert isinstance(viewpoint, str), 'Viewpoint must be a string'
            assert len(viewpoint) > 0, 'Viewpoint cannot be empty'
            assert viewpoint in const.YAWALIAS, (
                'Invalid viewpoint provided.  Must be one of: %s'
                % (list(const.YAWALIAS.keys()),)
            )

        if name is not None:
            nid = _ensure_names(ibs, name)
        else:
            nid = None
    except AssertionError as ex:
        raise controller_inject.WebInvalidInput(str(ex), parameter)

    gid = _ensure_images(ibs, image)

    gid_list = [gid]
    bbox_list = [bbox]
    theta_list = None if theta is None else [theta]
    species_list = None if species is None else [species]
    viewpoint_list = None if viewpoint is None else [viewpoint]
    nid_list = None if nid is None else [nid]

    aid_list = ibs.add_annots(
        gid_list,
        bbox_list=bbox_list,
        theta_list=theta_list,
        species_list=species_list,
        viewpoint_list=viewpoint_list,
        nid_list=nid_list,
    )
    assert len(aid_list) == 1
    aid = aid_list[0]

    response = _annotation(ibs, aid)
    print(response)
    return response


@register_api(_prefix('name'), methods=['POST'])
def microsoft_name_add(ibs, alias, *args, **kwargs):
    r"""
    Add a Name to the system and return the UUID
    ---
    parameters:
    - name: alias
      in: formData
      description: a user-defined string to specify an alias for this name, useful for forward facing interfaces
      required: false
      type: string
    consumes:
    - multipart/form-data
    produces:
    - application/json
    responses:
      200:
        description: Returns an Name model with a UUID
        schema:
          $ref: "#/definitions/Name"
      400:
        description: Invalid input parameter
    """
    try:
        parameter = 'alias'
        assert isinstance(alias, str), 'Alias must be a string'
        assert len(alias) > 0, 'Alias cannot be empty'
    except AssertionError as ex:
        raise controller_inject.WebInvalidInput(str(ex), parameter)

    nid = ibs.add_names(alias)
    return _name(ibs, nid)


@register_api(_prefix('image/annotations'), methods=['GET'], __api_plural_check__=False)
def microsoft_image_annotations(ibs, image, *args, **kwargs):
    r"""
    Get the Annotation models for a given Image model
    ---
    parameters:
    - name: image
      in: body
      description: A Image model to use for querying associated Annotation UUIDs
      required: true
      schema:
        $ref: "#/definitions/Image"
    produces:
    - application/json
    responses:
      200:
        description: Returns a (possibly empty) list of Annotations models
        schema:
          type: array
          items:
            $ref: "#/definitions/Annotation"
      400:
        description: Invalid input parameter
    """
    gid = _ensure_images(ibs, image)
    aid_list = ibs.get_image_aids(gid)

    annotation_list = [_annotation(ibs, aid) for aid in aid_list]
    response = {
        'annotations': annotation_list,
    }
    return response


@register_api(_prefix('name/annotations'), methods=['GET'], __api_plural_check__=False)
def microsoft_name_annotations(ibs, name, *args, **kwargs):
    r"""
    Get the Annotation models for a given Name model
    ---
    parameters:
    - name: name
      in: body
      description: A Name model to use for querying associated Annotation UUIDs
      required: true
      schema:
        $ref: "#/definitions/Name"
    produces:
    - application/json
    responses:
      200:
        description: Returns a (possibly empty) list of Annotations models
        schema:
          type: array
          items:
            $ref: "#/definitions/Annotation"
      400:
        description: Invalid input parameter
    """
    nid = _ensure_names(ibs, name)
    aid_list = ibs.get_name_aids(nid)

    annotation_list = [_annotation(ibs, aid) for aid in aid_list]
    response = {
        'annotations': annotation_list,
    }
    return response


@register_api(_prefix('annotation'), methods=['GET'])
def microsoft_annotation_metadata(ibs, annotation, *args, **kwargs):
    r"""
    Get the Annotation's metadata (bbox, theta, species, viewpoint).

    We provide no setters for Annotations.  If you wish to modify the metadata on an Annotation model, please delete it and add it again.  Modifying the metadata on an annotation may alter its derived UUID
    ---
    parameters:
    - name: annotation
      in: body
      description: A Annotation model
      required: true
      schema:
        $ref: "#/definitions/Image"
    produces:
    - application/json
    responses:
      200:
        description: Returns a JSON object with the metadata for the annotation
        schema:
          type: object
      400:
        description: Invalid input parameter
    """
    aid = _ensure_annotations(ibs, annotation)
    gid = ibs.get_annot_gids(aid)
    image = _image(ibs, gid)
    nid = ibs.get_annot_nids(aid)
    if nid is None or nid <= 0:
        name = None
    else:
        name = _name(ibs, nid)

    metadata = {
        'image': image,
        'name': name,
        'bbox': ibs.get_annot_bboxes(aid),
        'theta': ibs.get_annot_thetas(aid),
        'species': ibs.get_annot_species_texts(aid),
        'viewpoint': ibs.get_annot_viewpoints(aid),
    }
    return metadata


@register_api(_prefix('detect'), methods=['GET'])
def microsoft_detect_model(ibs, *args, **kwargs):
    r"""
    Returns the available models and their supported species for detection.  These models are pre-trained and are downloaded as needed on first start-up.

    The current model names that are supported for demoing and their species:
    - hammerhead -> shark_hammerhead
    - jaguar     -> jaguar
    - lynx       -> lynx
    - manta      -> manta_ray_giant
    - seaturtle  -> fish, ignore, person, turtle_green, turtle_green+head, turtle_hawksbill, turtle_hawksbill+head
    - ggr2       -> giraffe, zebra
    ---
    produces:
    - application/json
    responses:
      200:
        description: Returns an object of models (keys) and their supported species (values) listed as an array of strings.
        schema:
          type: object
    """
    hidden_models = [
        'hendrik_elephant',
        'hendrik_elephant_ears',
        'hendrik_elephant_ears_left',
        'hendrik_dorsal',
        'candidacy',
        None,
    ]
    return ibs.models_cnn_lightnet(hidden_models=hidden_models)


def microsoft_detect_input_validation(model, score_threshold, use_nms, nms_threshold):
    from wbia.algo.detect.lightnet import CONFIG_URL_DICT

    try:
        parameter = 'model'
        assert model in CONFIG_URL_DICT, 'Specified model is not supported'

        parameter = 'score_threshold'
        assert isinstance(score_threshold, float), 'Score threshold must be a float'
        assert (
            0.0 <= score_threshold and score_threshold <= 1.0
        ), 'Score threshold is invalid, must be in range [0.0, 1.0]'

        parameter = 'use_nms'
        assert isinstance(use_nms, bool), 'NMS flag must be a boolean'

        parameter = 'nms_threshold'
        assert isinstance(nms_threshold, float), 'NMS threshold must be a float'
        assert (
            0.0 <= nms_threshold and nms_threshold <= 1.0
        ), 'NMS threshold is invalid, must be in range [0.0, 1.0]'
    except AssertionError as ex:
        raise controller_inject.WebInvalidInput(str(ex), parameter)


@register_ibs_method
def microsoft_detect(
    ibs,
    images,
    model,
    score_threshold=0.0,
    use_nms=True,
    nms_threshold=0.4,
    __jobid__=None,
    *args,
    **kwargs,
):
    depc = ibs.depc_image

    gid_list = _ensure_images(ibs, images, unpack=False)

    try:
        config = {
            'algo': 'lightnet',
            'config_filepath': model,
            'weight_filepath': model,
            'sensitivity': score_threshold,
            'nms': use_nms,
            'nms_thresh': nms_threshold,
        }
        results_list = depc.get_property('localizations', gid_list, None, config=config)

        # DEPRICATE
        #     # Do not commit results, return them outright
        #     aids_list = ibs.commit_localization_results(gid_list, results_list)
    except Exception:
        print(str(traceback.format_exc()))
        raise controller_inject.WebException(
            'Detection process failed for an unknown reason'
        )

    # DEPRICATE
    #     # Do not commit detections
    #     detections_list = {
    #         'detections_list': [
    #             {
    #                 'detections' : [
    #                     _detection(ibs, aid)
    #                     for aid in aid_list
    #                 ],
    #             }
    #             for aid_list in aids_list
    #         ]
    #     }

    zipped = zip(gid_list, results_list)
    detections_list = {
        'detections_list': [
            {
                'detections': [
                    _detection(ibs, gid, result) for result in zip(*result_list[1:])
                ],
            }
            for gid, result_list in zipped
        ]
    }

    return detections_list


@register_api(_prefix('detect'), methods=['POST'])
def microsoft_detect_upload(
    ibs, model, score_threshold=0.0, use_nms=True, nms_threshold=0.4, *args, **kwargs
):
    r"""
    Returns the detection results for an uploaded image and a provided model configuration.

    The uploaded image will be used to perform object detection for a given model.  A detection will return at most 845 Detections (13 x 13 grid of 5 bounding boxes based on the YOLO v2 by Redmon et al.).
    ---
    parameters:
    - name: image
      in: formData
      description: The image for which detection will be processed.
      required: true
      type: file
      enum:
      - image/png
      - image/jpg
      - image/tiff
    - name: model
      in: formData
      description: The model name to use with detection.  Available models can be queried using the API $PREFIX/detect/model/
      required: true
      type: string
    - name: score_threshold
      in: formData
      description: A score threshold to filter the detection results.  Default 0.0.  Must be in range [0, 1], inclusive
      required: false
      type: number
      format: float
    - name: use_nms
      in: formData
      description: Flag to turn on/off Non-Maximum Suppression (NMS).  Default True.  Must be in range [0, 1], inclusive
      required: false
      type: boolean
    - name: nms_threshold
      in: formData
      description: A Intersection-over-Union (IoU) threshold to suppress detection results with NMS.  Default 0.0.  Must be in range [0, 1], inclusive
      required: false
      type: number
      format: float
    consumes:
    - multipart/form-data
    responses:
      200:
        description: Returns an array of Detection models on the uploaded image
        schema:
          type: array
          items:
            $ref: "#/definitions/Detection"
      400:
        description: Invalid input parameter
      415:
        description: Unsupported media type in the request body. Currently only image/png, image/jpeg, image/tiff are supported.
    """
    ibs = current_app.ibs

    # Input argument validation
    image = microsoft_image_upload(ibs)
    microsoft_detect_input_validation(model, score_threshold, use_nms, nms_threshold)

    try:
        images = [image]
        detections_list = microsoft_detect(
            ibs, images, model, score_threshold, use_nms, nms_threshold
        )
    except Exception:
        print(str(traceback.format_exc()))
        raise controller_inject.WebException(
            'Detection process failed for an unknown reason'
        )

    detections_list = detections_list.get('detections_list')
    assert len(detections_list) == 1
    detections = detections_list[0]

    print(detections)
    return detections


@register_api(_prefix('detect/batch'), methods=['POST'])
def microsoft_detect_batch(
    ibs,
    images,
    model,
    score_threshold=0.0,
    use_nms=True,
    nms_threshold=0.4,
    async_=True,
    callback_url=None,
    callback_method=None,
    *args,
    **kwargs,
):
    r"""
    The asynchronous variant of POST 'detect' that takes in a list of Image models and returns a task ID.

    It may be more ideal for a particular application to upload many images at one time and perform processing later in a large batch.  This type of batch detection is certainly much more efficient because the detection on GPU can process more images in parallel.  However, if you intend to run the detector on an upload as quickly as possible, please use the on-demand, non-batched API.
    ---
    parameters:
    - name: images
      in: body
      description: A JSON list of Image models for which detection will be processed.
      required: true
      type: array
    - name: model
      in: formData
      description: The model name to use with detection.  Available models can be queried using the API $PREFIX/detect/model/
      required: true
      type: string
    - name: score_threshold
      in: formData
      description: A score threshold to filter the detection results.  Default 0.0.  Must be in range [0, 1], inclusive
      required: false
      type: number
      format: float
    - name: use_nms
      in: formData
      description: Flag to turn on/off Non-Maximum Suppression (NMS).  Default True.
      required: false
      type: boolean
    - name: nms_threshold
      in: formData
      description: A Intersection-over-Union (IoU) threshold to suppress detection results with NMS.  Default 0.0.  Must be in range [0, 1], inclusive
      required: false
      type: number
      format: float
    - name: callback_url
      in: formData
      description: The URL of where to callback when the task is completed, must be a fully resolvable address and accessible.  The callback will include a 'body' parameter called `task` which will provide a Task model
      required: false
      type: string
      format: url
    - name: callback_method
      in: formData
      description: The HTTP method for which to make the callback.  Default POST.
      required: false
      type: string
      enum:
      - get
      - post
      - put
      - delete
    consumes:
    - multipart/form-data
    responses:
      200:
        description: Returns a Task model
        schema:
          $ref: "#/definitions/Task"
      400:
        description: Invalid input parameter
      x-task-response:
        description: The task returns an array of arrays of Detection models, in parallel lists with the provided Image models
        schema:
          type: array
          items:
            schema:
              type: array
              items:
                $ref: "#/definitions/Detection"
    """
    ibs = current_app.ibs

    # Input argument validation
    for index, image in enumerate(images):
        try:
            parameter = 'images:%d' % (index,)
            assert 'uuid' in image, 'Image Model provided is invalid, missing UUID key'
        except AssertionError as ex:
            raise controller_inject.WebInvalidInput(str(ex), parameter)

    microsoft_detect_input_validation(model, score_threshold, use_nms, nms_threshold)

    try:
        parameter = 'async_'
        assert isinstance(async_, bool), 'Asynchronous flag must be a boolean'

        parameter = 'callback_url'
        assert callback_url is None or isinstance(
            callback_url, str
        ), 'Callback URL must be a string'
        if callback_url is not None:
            assert callback_url.startswith('http://') or callback_url.startswith(
                'https://'
            ), 'Callback URL must start with http:// or https://'

        parameter = 'callback_method'
        assert callback_method is None or isinstance(
            callback_method, str
        ), 'Callback URL must be a string'
        if callback_method is not None:
            callback_method = callback_method.lower()
            assert callback_method in [
                'get',
                'post',
                'put',
                'delete',
            ], 'Unsupported callback method, must be one of ("get", "post", "put", "delete")'
    except AssertionError as ex:
        raise controller_inject.WebInvalidInput(str(ex), parameter)

    args = (
        images,
        model,
    )
    kwargs = {
        'score_threshold': score_threshold,
        'use_nms': use_nms,
        'nms_threshold': nms_threshold,
    }

    if async_:
        taskid = ibs.job_manager.jobiface.queue_job(
            'microsoft_detect', callback_url, callback_method, *args, **kwargs
        )
        response = _task(ibs, taskid)
    else:
        response = ibs.microsoft_detect(*args, **kwargs)

    return response


@register_api(_prefix('identify'), methods=['POST'])
def microsoft_identify(
    ibs,
    query_annotation,
    database_annotations,
    algorithm,
    callback_url=None,
    callback_method=None,
    *args,
    **kwargs,
):
    r"""
    The asynchronous call to identify a list of pre-uploaded query annotations against a database of annotations.  Returns a task ID.

    This call expects all of the data has been already uploaded and the
    appropriate metadata set.  For example, this call expects any known
    ground-truth nameshave already been made and assigned to the appropriate
    Annotation models.  Conversely, we also provide a method for marking individual
    reviews between two Annotation models.  The decision between a given pair
    should always be one of ['match', 'nomatch', or 'notcomp'].  The full docuemntation
    for this call can be seen in the POST 'decision' API.
    ---
    parameters:
    - name: query_annotation
      in: formData
      description: An Annotation models to query for an identification
      required: true
      schema:
        $ref: "#/definitions/Annotation"
    - name: database_annotations
      in: formData
      description: A JSON list of Annotation models to compare the query Annotation against
      required: true
      type: array
      items:
        $ref: "#/definitions/Annotation"
    - name: algorithm
      in: formData
      description: The algorithm you with to run ID with.  Must be one of "HotSpotter", "CurvRank", "Finfindr", or "Deepsense"
      required: true
      type: string
    - name: callback_url
      in: formData
      description: The URL of where to callback when the task is completed, must be a fully resolvable address and accessible.  The callback will include a 'body' parameter called `task` which will provide a Task model
      required: false
      type: string
      format: url
    - name: callback_method
      in: formData
      description: The HTTP method for which to make the callback.  Default POST.
      required: false
      type: string
      enum:
      - get
      - post
      - put
      - delete
    consumes:
    - multipart/form-data
    responses:
      200:
        description: Returns a Task model
        schema:
          $ref: "#/definitions/Task"
      400:
        description: Invalid input parameter
    """
    qaid = _ensure_annotations(ibs, query_annotation)
    daid_list = _ensure_annotations(ibs, database_annotations)

    try:
        parameter = 'database_annotations'
        assert (
            len(daid_list) > 0
        ), 'Cannot specify an empty list of database Annotations to compare against'

        parameter = 'algorithm'
        assert isinstance(algorithm, str), 'Must specify the algorithm as a string'
        algorithm = algorithm.lower()
        assert algorithm in [
            'hotspotter',
            'curvrank',
            'deepsense',
            'finfindr',
            'kaggle7',
            'kaggleseven',
        ], 'Must specify the algorithm for ID as HotSpotter, CurvRank, Deepsense, Finfindr, Kaggle7'

        parameter = 'callback_url'
        assert callback_url is None or isinstance(
            callback_url, str
        ), 'Callback URL must be a string'
        if callback_url is not None:
            assert callback_url.startswith('http://') or callback_url.startswith(
                'https://'
            ), 'Callback URL must start with http:// or https://'

        parameter = 'callback_method'
        assert callback_method is None or isinstance(
            callback_method, str
        ), 'Callback URL must be a string'
        if callback_method is not None:
            callback_method = callback_method.lower()
            assert callback_method in [
                'get',
                'post',
                'put',
                'delete',
            ], 'Unsupported callback method, must be one of ("get", "post", "put", "delete")'
    except AssertionError as ex:
        raise controller_inject.WebInvalidInput(str(ex), parameter)

    if algorithm in ['hotspotter']:
        query_config_dict = {}
    elif algorithm in ['curvrank']:
        query_config_dict = {
            'pipeline_root': 'CurvRankFluke',
        }
    elif algorithm in ['deepsense']:
        query_config_dict = {
            'pipeline_root': 'Deepsense',
        }
    elif algorithm in ['finfindr']:
        query_config_dict = {
            'pipeline_root': 'Finfindr',
        }
    elif algorithm in ['kaggle7', 'kaggleseven']:
        query_config_dict = {
            'pipeline_root': 'KaggleSeven',
        }

    user_feedback = {
        'aid1': [],
        'aid2': [],
        'p_match': [],
        'p_nomatch': [],
        'p_notcomp': [],
    }
    echo_query_params = False
    args = (
        [qaid],
        daid_list,
        user_feedback,
        query_config_dict,
        echo_query_params,
    )
    kwargs = {
        'endpoint': url_for('microsoft_identify_visualize'),
        'n': 5,
    }
    taskid = ibs.job_manager.jobiface.queue_job(
        'query_chips_graph_microsoft', callback_url, callback_method, *args, **kwargs
    )

    return _task(ibs, taskid)


@register_ibs_method
def query_chips_graph_microsoft(ibs, *args, **kwargs):
    endpoint = kwargs.pop('endpoint')
    result = ibs.query_chips_graph(*args, **kwargs)

    cm_dict = result['cm_dict']
    cm_key_list = list(cm_dict.keys())
    assert len(cm_key_list) == 1
    cm_key = cm_key_list[0]
    cm = cm_dict[cm_key]

    qaid = ibs.get_annot_aids_from_uuid(cm['qannot_uuid'])
    reference = cm['dannot_extern_reference']

    response = {
        'annotations': [],
        'names': [],
    }

    zipped = list(
        zip(cm['annot_score_list'], cm['dannot_uuid_list'], cm['dannot_extern_list'])
    )
    zipped = sorted(zipped, reverse=True)
    for dannot_score, dannot_uuid, dannot_extern in zipped:
        daid = ibs.get_annot_aids_from_uuid(dannot_uuid)
        annotation = _annotation(ibs, daid)
        extern_url_postfix = None
        if dannot_extern:
            query_annotation_ = ut.to_json(_annotation(ibs, qaid)).replace(' ', '')
            database_annotation_ = ut.to_json(_annotation(ibs, daid)).replace(' ', '')
            args = (
                endpoint,
                reference,
                query_annotation_,
                database_annotation_,
            )
            extern_url_postfix = (
                '%s?reference=%s&query_annotation=%s&database_annotation=%s' % args
            )
        response['annotations'].append(
            {
                'score': dannot_score,
                'annotation': annotation,
                'visualize': extern_url_postfix,
            }
        )

    zipped = list(zip(cm['name_score_list'], cm['unique_name_list']))
    zipped = sorted(zipped, reverse=True)
    for name_score, name_text in zipped:
        if name_text == const.UNKNOWN:
            continue
        nid = ibs.get_name_rowids_from_text(name_text)
        response['names'].append({'score': name_score, 'name': _name(ibs, nid)})

    return response


@register_route(_prefix('visualize'), methods=['GET'])
def microsoft_identify_visualize(
    reference, query_annotation, database_annotation, version='heatmask'
):
    r"""
    Visualize the results of matching, precomputed and rendered to disk for easy look-up.

    ---
    parameters:
    - name: reference
      in: formData
      description: A reference string to refer to the rendered match results on disk for a given identification Task
      required: true
      type: string
    - name: query_annotation
      in: formData
      description: The query Annotation model
      required: true
      schema:
        $ref: "#/definitions/Annotation"
    - name: database_annotation
      in: formData
      description: The database Annotation model
      required: true
      schema:
        $ref: "#/definitions/Annotation"
    - name: version
      in: formData
      description: The version of the visualization.  Must be one of "heatmask" or "original".  Defaults to "heatmask"
      required: false
      type: string
    consumes:
    - multipart/form-data
    responses:
      200:
        description: Returns a Task model
        schema:
          $ref: "#/definitions/Task"
      400:
        description: Invalid input parameter
    """
    ibs = current_app.ibs

    from wbia.web.apis_query import query_chips_graph_match_thumb

    qaid = _ensure_annotations(ibs, query_annotation)
    daid = _ensure_annotations(ibs, database_annotation)
    quuid, duuid = ibs.get_annot_uuids([qaid, daid])
    quuid = str(quuid)
    duuid = str(duuid)
    response = query_chips_graph_match_thumb(
        extern_reference=reference,
        query_annot_uuid=quuid,
        database_annot_uuid=duuid,
        version=version,
    )
    return response


@register_api(_prefix('task'), methods=['GET'])
def microsoft_task_status(ibs, task):
    r"""
    Check the status of an asynchronous Task.

    A Task is an asynchronous task that was launched as a background process with an optional callback.  The status of a given Task with a UUID can be checked with this call.  The status of the call depends on where in the execution queue the Task is currently, which will be processed in a first-come-first-serve list and only one Task at a time to present atomicity of the API.

    The status can be one of the following:
    - received   -> The Task request was received but has not passed any input validation.
    - accepted   -> The Task request has passed basic input validation and will be queued soon for execution.
    - queued     -> The Task is queued in the execution list and will be processed in order and one at a time.
    - working    -> The Task is being processed, awaiting completed results or an error exception
    - publishing -> The Task is publishing the results of the background API call.
    - completed  -> One of two end states: the Task is complete, completed results available for downloading with the REST API.
    - exception  -> One of two end states: the Task has encountered an error, an error message can be received using the results REST API.
    - unknown    -> The Task you asked for is not known, indicating that the either UUID is not recognized (i.e. a Task with that ID was never currently created) or the server has been restarted.

    **Important: when the API server is restarted, all queued and running background Tasks are killed and all Task requests and cached results are deleted.**
    ---
    parameters:
    - name: task
      in: body
      description: A Task model
      required: true
      schema:
        $ref: "#/definitions/Task"
    responses:
      200:
        description: Returns the status of the provided Task
        schema:
          type: object
          properties:
            status:
              type: string
              enum:
              - received
              - accepted
              - queued
              - working
              - publishing
              - completed
              - exception
              - unknown
      400:
        description: Invalid input parameter
    """
    ibs = current_app.ibs

    # Input argument validation
    try:
        parameter = 'task'
        assert 'uuid' in task, 'Task Model provided is invalid, missing UUID key'
    except AssertionError as ex:
        raise controller_inject.WebInvalidInput(str(ex), parameter)

    uuid_ = task.get('uuid', None)
    assert uuid_ is not None
    status = ibs.get_job_status(uuid_)
    status = status.get('jobstatus', None)
    return {'status': status}


@register_api(_prefix('task'), methods=['POST'])
def microsoft_task_result(ibs, task):
    r"""
    Retrieve the result of a completed asynchronous Task
    ---
    parameters:
    - name: task
      in: body
      description: A Task model
      required: true
      schema:
        $ref: "#/definitions/Task"
    responses:
      200:
        description: Returns the result of the provided Task
      400:
        description: Invalid input parameter
    """
    ibs = current_app.ibs

    # Input argument validation
    try:
        parameter = 'task'
        assert 'uuid' in task, 'Task Model provided is invalid, missing UUID key'
    except AssertionError as ex:
        raise controller_inject.WebInvalidInput(str(ex), parameter)

    uuid_ = task.get('uuid', None)
    assert uuid_ is not None
    result = ibs.get_job_result(uuid_)
    result = result.get('json_result', None)
    return result


@register_api(_prefix('image'), methods=['DELETE'])
def microsoft_image_delete(ibs, image):
    r"""
    Delete an Image model from the database.

    Also deletes any derived Detection models or any added Annotations for the deleted Image model.
    ---
    parameters:
    - name: image
      in: body
      description: A Image model to delete
      required: true
      schema:
        $ref: "#/definitions/Image"
    produces:
    - application/json
    responses:
      200:
        description: Returns a success status flag (True if deleted)
        schema:
          type: boolean
      400:
        description: Invalid input parameter

    """
    ibs = current_app.ibs

    # Input argument validation
    gid = _ensure_images(ibs, image)
    ibs.delete_images(gid)

    return {'status': 'deleted'}


@register_api(_prefix('annotation'), methods=['DELETE'])
def microsoft_annotation_delete(ibs, annotation):
    r"""
    Delete an Annotation model from the database.
    ---
    parameters:
    - name: annotation
      in: body
      description: A Annotation model to delete
      required: true
      schema:
        $ref: "#/definitions/Annotation"
    produces:
    - application/json
    responses:
      200:
        description: Returns a success status flag (True if deleted)
        schema:
          type: boolean
      400:
        description: Invalid input parameter

    """
    ibs = current_app.ibs

    # Input argument validation
    aid = _ensure_annotations(ibs, annotation)
    ibs.delete_annots(aid)

    return {'status': 'deleted'}


@register_api(_prefix('name'), methods=['DELETE'])
def microsoft_name_delete(ibs, name):
    r"""
    Delete a Name model from the database.

    This operation DOES NOT delete any associated Annotation models with this Name, it simply disassociates them.
    ---
    parameters:
    - name: name
      in: body
      description: A Name model to delete
      required: true
      schema:
        $ref: "#/definitions/Name"
    produces:
    - application/json
    responses:
      200:
        description: Returns a success status flag (True if deleted)
        schema:
          type: boolean
      400:
        description: Invalid input parameter

    """
    ibs = current_app.ibs

    # Input argument validation
    nid = _ensure_names(ibs, name)
    ibs.delete_names(nid)

    return {'status': 'deleted'}


@register_api(_prefix('test/annotations'), methods=['GET'], __api_plural_check__=False)
def microsoft_get_test_data(ibs, dataset):
    r"""
    Return test data annotations.
    ---
    parameters:
    - name: dataset
      in: body
      description: A name for the dataset.  Must be one of 'zebra', 'dolphin', 'humpback'
      required: true
      type: string
    produces:
    - application/json
    responses:
      200:
        description: A JSON object with the query and database Annotation model lists
      400:
        description: Invalid input parameter
    """
    response = ibs.api_test_datasets_id(dataset)

    for key in response:
        aid_list = response[key]
        response[key] = [_annotation(ibs, aid) for aid in aid_list]

    return response


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.web.app
        python -m wbia.web.app --allexamples
        python -m wbia.web.app --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
