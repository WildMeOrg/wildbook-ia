function monitorJob(jobid, callback, index, progress) {

    var progressBar = $('#progress-bar-' + index + '-' + progress);

    $.ajax({
        url: "/api/engine/job/status/?jobid=" + jobid,
        method: "GET",
        contentType: "application/json; charset=utf-8",
        dataType: "json",
        success: function(response) {

            status = response.response.jobstatus

            // console.log('job: ' + jobid + ' (status: ' + status + ')')
            if (status != 'completed') {
              setTimeout(function() {
                monitorJob(jobid, callback, index);
              }, 500);
            } else {

              $.ajax({
                  url: "/api/engine/job/result/?jobid=" + jobid,
                  method: "GET",
                  contentType: "application/json; charset=utf-8",
                  dataType: "json",
                  success: function(response) {
                      response = response.response.json_result
                      progressBar.css({"width": "90%"});
                      callback(index, response)
                  },
                  error: function(response) {
                    submitError(index, progress)
                  },
              });

            }
        },
        error: function(response) {
          submitError(index, progress)
        },
    });
}

function retrieveIdentification(index, response) {
    var progressBar = $('#progress-bar-' + index + '-3');

    name = response

    // $('img#match-' + index).attr('src', response.match).on('load', function() {
    //   progressBar.removeClass('progress-bar-striped')
    //   progressBar.css({"width": "100%"});
    // })

    if (name == 'Zak') {
      species = 'Chapman’s Zebra'
      viewpoint = 'Right side'
      sex = 'Male'
      age = '19 years old'
    } else if (name == 'Florence') {
      species = 'Grant’s Zebra'
      viewpoint = 'Right side'
      sex = 'Female'
      age = '9 years old'
    } else if (name == 'Pete') {
      species = 'Grant’s Zebra'
      viewpoint = 'Right side'
      sex = 'Male'
      age = '3 years old'
    } else {
      name = 'Unknown'
      species = registry[index].detection.species2
      viewpoint = registry[index].detection.viewpoint2 + ' side'
      sex = 'Unknown'
      age = 'Unknown'
    }

    registry[index].detection.name = name
    registry[index].detection.species = species
    registry[index].detection.viewpoint = viewpoint
    registry[index].detection.sex = sex
    registry[index].detection.age = age

    console.log(registry[index])
    console.log(registry[index].detection)

    $('#id-container-labels-' + index).html(
      'Name:<br/>' +
      'Species:<br/>' +
      'Viewpoint:<br/>' +
      'Sex:<br/>' +
      'Age:<br/>'
    )

    $('#id-container-values-' + index).html(
      '' + name + '<br/>' +
      '' + species + '<br/>' +
      '' + viewpoint + '<br/>' +
      '' + sex + '<br/>' +
      '' + age
    )

    progressBar.removeClass('progress-bar-striped')
    progressBar.css({"width": "100%"});
}

function submitIdentification() {

    var index = queue.identify.shift()

    if (index === undefined) {
      return
    }

    var progressBar = $('#progress-bar-' + index + '-3');
    progressBar.css({"width": "10%"});

    var record = registry[index]

    $('img#match-' + index).attr('src', '/static/images/loading.gif')

    var data = {
      'annot_uuid': record.detection.uuid,
    }

    $.ajax({
        url: "/api/engine/review/query/chip/best/",
        method: "POST",
        contentType: "application/json; charset=utf-8",
        dataType: "json",
        data: JSON.stringify(data),
        success: function(response) {

            jobid = response.response
            registry[index].identify_job = jobid

            progressBar.css({"width": "30%"});
            monitorJob(jobid, retrieveIdentification, index, 3)
        },
        error: function(response) {
          submitError(index, 3)
        },
    });

    submitIdentification()
}


function retrieveClassification(index, response) {
    var progressBar = $('#progress-bar-' + index + '-2');

    response = response[0]

    registry[index].detection.species2 = response.species
    registry[index].detection.viewpoint2 = response.viewpoint

    progressBar.removeClass('progress-bar-striped')
    progressBar.css({"width": "100%"});

    queue.identify.push(index)
    submitIdentification()
}

function submitClassification() {

    var index = queue.classify.shift()

    if (index === undefined) {
      return
    }

    var progressBar = $('#progress-bar-' + index + '-2');
    progressBar.css({"width": "10%"});

    var record = registry[index]

    var data = {
      'annot_uuid_list': [record.detection.uuid],
    }

    $.ajax({
        url: "/api/engine/labeler/cnn/",
        method: "POST",
        contentType: "application/json; charset=utf-8",
        dataType: "json",
        data: JSON.stringify(data),
        success: function(response) {

            jobid = response.response
            registry[index].classify_job = jobid

            progressBar.css({"width": "30%"});
            monitorJob(jobid, retrieveClassification, index, 2)
        },
        error: function(response) {
          submitError(index, 2)
        },
    });

    submitClassification()
}

function retrieveDetection(index, response) {
    var progressBar = $('#progress-bar-' + index + '-1');

    detections = response.results_list[0]
    registry[index].detections = detections

    bestDetection = null
    bestConfidence = 0.0
    for(var i = 0; i < registry[index].detections.length; i++) {
      detection = registry[index].detections[i]

      if(detection.confidence > bestConfidence) {
        bestDetection = detection
        bestConfidence = detection.confidence
      }
    }

    registry[index].detection = bestDetection

    if (bestDetection === null) {
      progressBar.removeClass('progress-bar-striped')
      progressBar.css({"width": "100%"});

      $("img#annot-" + index).fadeOut(1000)
      $("#row-container-2" + index).fadeOut(1000)
      $("#row-container-3" + index).fadeOut(1000)

      return
    }

    url = '/api/annot/src/' + bestDetection.id
    $('img#annot-' + index).attr('src', url).on('load', function() {
      progressBar.removeClass('progress-bar-striped')
      progressBar.css({"width": "100%"});

      queue.classify.push(index)
      submitClassification()
    })
}

function submitDetection() {

    var index = queue.detect.shift()

    if (index === undefined) {
      return
    }

    var progressBar = $('#progress-bar-' + index + '-1');
    progressBar.css({"width": "10%"});

    var record = registry[index]

    $('img#annot-' + index).attr('src', '/static/images/loading.gif')

    $.ajax({
        url: "/api/image/uuid/?gid_list=[" + record.gid + "]",
        method: "GET",
        contentType: "application/json; charset=utf-8",
        dataType: "json",
        success: function(response) {

          uuid = response.response[0]
          registry[index].uuid = uuid

          var data2 = {
            'image_uuid_list': [uuid],
            // 'model_tag': 'candidacy',
          }

          $.ajax({
              url: "/api/engine/detect/cnn/yolo/",
              method: "POST",
              contentType: "application/json; charset=utf-8",
              dataType: "json",
              data: JSON.stringify(data2),
              success: function(response) {

                  jobid = response.response
                  registry[index].detect_job = jobid

                  progressBar.css({"width": "30%"});
                  monitorJob(jobid, retrieveDetection, index, 1)
              },
              error: function(response) {
                submitError(index, 1)
              },
          });

          progressBar.css({"width": "10%"});
        },
        error: function(response) {
          submitError(index, 1)
        },
    });

    submitDetection()
}

function submitUpload() {

    var index = queue.upload.shift()

    if (index === undefined) {
      return
    }

    var progressBar = $('#progress-bar-' + index + '-0');

    var data = new FormData();

    var record = registry[index]
    data.append("image", record.file);

    $('img#image-' + index).attr('src', '/static/images/loading.gif')

    $.ajax({
        xhr: function() {
            var xhr = new window.XMLHttpRequest();

            if (xhr.upload) {
                xhr.upload.addEventListener("progress", function(event) {
                  if (event.lengthComputable) {
                    var percent = 0;
                    var position = event.loaded || event.position;
                    var total    = event.total;
                        percent = Math.ceil(position / total * 90);

                    // Set the progress bar.
                    progressBar.css({"width": percent + "%"});
                  }
                }, false)
            }
            return xhr;
        },
        url: "/api/upload/image/",
        method: "POST",
        contentType: "application/json; charset=utf-8",
        dataType: "json",
        contentType: false,
        processData: false,
        cache: false,
        data: data,
        success: function(response) {
            gid = response.response
            registry[index].gid = gid

            // Load image
            url = '/api/image/src/' + gid
            $('img#image-' + index).attr('src', url).on('load', function() {
              progressBar.removeClass('progress-bar-striped')
              progressBar.css({"width": "100%"});

              queue.detect.push(index)
              submitDetection()
            })

        },
        error: function(response) {
          submitError(index, 0)
        },
    });

    submitUpload()
}


function submitError(index, task) {
  console.log('ERROR on index ' + index + " task " + task)
}

function registerFiles(files) {

    if (! initialized) {

      positionLogoUpload();

      setTimeout(function() {
        registerFiles(files);
      }, 800);

      initialized = true;

      return
    }

    var container = $('#images-container');

    // Add them to the pending files list.
    for (var i = 0; i < files.length; i++) {
        var file = files[i]
        var index = registry.length

        var row = $('<div class="row" id="row-' + index + '"></div>')
        container.append(row)

        var left    = $('<div class="col-lg-2 col-md-2 col-sm-2 col-xs-2"></div>')
        var element = $('<div class="col-lg-8 col-md-8 col-sm-8 col-xs-8 element" id="element-' + index + '"></div>')
        var right   = $('<div class="col-lg-2 col-md-2 col-sm-2 col-xs-2"></div>')

        row.append(left);
        row.append(element);
        row.append(right);

        var row2 = $('<div class="row" id="row-' + index + '"></div>')
        element.append(row2)

        var left2   = $('<div class="col-lg-4 col-md-4 col-sm-4 col-xs-4 element-left"></div>')
        var center2 = $('<div class="col-lg-4 col-md-4 col-sm-4 col-xs-4 element-center"></div>')
        var right2  = $('<div class="col-lg-4 col-md-4 col-sm-4 col-xs-4 element-right"></div>')

        row2.append(left2)
        row2.append(center2)
        row2.append(right2)

        var image = $('<img id="image-' + index + '" src="">');
        left2.append(image)

        var annot = $('<img id="annot-' + index + '" src="">');
        center2.append(annot)

        var texts = ['Upload', 'Detect', 'Classify', 'Identify']
        for (var i2 = 0; i2 < texts.length; i2++) {
          right2.append(
            '<div id="row-container-' + i2 + '" class="row">' +
              '<div class="col-lg-3 col-md-3 col-sm-12 col-xs-12 text-container">' + texts[i2] + '</div>' +
              '<div class="col-lg-9 col-md-9 col-sm-12 col-xs-12 progress-container">' +
                '<div id="progress" class="progress">' +
                  '<div id="progress-bar-' + index + '-' + i2 + '" class="progress-bar progress-bar' + i2 + ' progress-bar-striped active" role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100" style="width: 0%"></div>' +
                '</div>' +
              '</div>' +
            '</div>'
          )
        }

        right2.append(
            '<div id="id-container-labels-' + index + '" class="col-lg-6 col-md-6 col-sm-6 col-xs-6 id-container-labels"></div>' +
            '<div id="id-container-values-' + index + '" class="col-lg-6 col-md-6 col-sm-6 col-xs-6 id-container-values"></div>'
          )

        // var center3 = $('<div class="col-lg-12 col-md-12 col-sm-12 col-xs-12 element-center"></div>')
        // row2.append(center3)

        // var match = $('<img id="match-' + index + '" src="">');
        // center3.append(match)

        var record = {
          index: index,
          file: file,
          element: element,
        }

        registry.push(record)
        queue.upload.push(index)
    }

    submitUpload();
}

function positionLogoInit() {
    $("#dropbox-text").hide()
    $("#progress").hide()
    $("#logo").hide()

    $('#dropbox-container').removeClass("hidden")
    $('#dropbox-container').css({'height': window.innerHeight})

    offset = (window.innerHeight - $("#dropbox").height()) * 0.5
    $('#dropbox-container-inside').css('margin-top', offset + "px")
    $("#dropbox-container-inside").hide().fadeIn(1200)

    animations.dropbox = setTimeout(function() {
      $("#dropbox-text").fadeIn(1200)
      $("#logo").fadeIn(1200)
    }, 1000);
}

function positionLogoUpload() {

  if (animations.dropbox !== undefined) {
    clearTimeout(animations.dropbox)
  }

  $('#dropbox-text').animate({
    'opacity': '0',
    'height': '0px',
  }, 300)

  $('#dropbox-container-inside').animate({
    'margin-top': "10px",
    'margin-bottom': "10px",
  }, 700);

  $('#dropbox').animate({
    'max-width': "100px",
    'padding': '4px',
  }, 700);

  $('#progress').delay(300).fadeIn(1000)
}

function init(image1, image2) {
    var $dropbox = $("#dropbox");
    var $logo    = $("#logo");

    $dropbox.attr('src', image1).on('load', function() {
      $logo.attr('src', image2).on('load', function() {
          positionLogoInit();
      });
    });

    $dropbox.on("dragover", function(event) {
        event.stopPropagation();
        event.preventDefault();
    });

    $dropbox.on("dragenter", function(event) {
        event.stopPropagation();
        event.preventDefault();
        $(this).addClass("active");
    });

    $dropbox.on("dragleave", function(event) {
        event.stopPropagation();
        event.preventDefault();
        $(this).removeClass("active");
    });

    $dropbox.on("drop", function(event) {
        event.preventDefault();
        $(this).removeClass("active");

        registerFiles(event.originalEvent.dataTransfer.files);
    });

    function stopDefault(event) {
        event.stopPropagation();
        event.preventDefault();
    }

    $(document).on("dragenter", stopDefault);
    $(document).on("dragover",  stopDefault);
    $(document).on("drop",      stopDefault);
}
