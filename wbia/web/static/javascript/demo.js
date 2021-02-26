function getRandomInt(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min)) + min; //The maximum is exclusive and the minimum is inclusive
}

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
              }, getRandomInt(1000, 3001));
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

function getValue(url, callback, index) {

    $.ajax({
        url: url,
        method: "GET",
        contentType: "application/json; charset=utf-8",
        dataType: "json",
        success: function(response) {
            response = response.response
            callback(response, index)
        }
    });
}

function retrieveIdentification(index, response) {
    var progressBar = $('#progress-bar-' + index + '-3');

    // $('img#match-' + index).attr('src', response.match).on('load', function() {
    //   progressBar.removeClass('progress-bar-striped')
    //   progressBar.css({"width": "100%"});
    // })

    console.log(response)

    registry[index].detection.name   = response.name
    registry[index].detection.sex    = response.sex
    registry[index].detection.age    = response.age
    registry[index].detection.extern = response.extern

    $('#id-container-labels-' + index).html(
      'Name:<br/>' +
      'Species:<br/>' +
      'Viewpoint:<br/>' +
      'Sex:<br/>' +
      'Age:<br/>'
    )

    $('#id-container-values-' + index).html(
      '' + registry[index].detection.name + '<br/>' +
      '' + registry[index].detection.species + '<br/>' +
      '' + registry[index].detection.viewpoint + '<br/>' +
      '' + registry[index].detection.sex + '<br/>' +
      '' + registry[index].detection.age + '<br/>' +
      '<a class="btn btn-default btn-xs" role="button" data-toggle="collapse" style="margin-top: 3px;" id="button-' + index + '" href="#collapse-' + index + '" aria-expanded="false" aria-controls="collapse-' + index + '">Evidence</a>'
    )

    extern = registry[index].detection.extern
    match_url = '/api/query/graph/match/thumb/?extern_reference=' + extern.reference + '&query_annot_uuid=' + extern.qannot_uuid + '&database_annot_uuid=' + extern.dannot_uuid

    registry[index].detection.match = {
      'clean': match_url + '&version=clean',
      'dirty': match_url + '&version=heatmask',
    }

    match = registry[index].detection.match
    $evidence = $('img.evidence#evidence-' + index)

    // Download both clean and dirty images, set to src (clean last)
    $evidence.attr('src', match.dirty).on('load', function() {
      $evidence.attr('src', match.clean).on('load', function() {
        $('a#button-' + index).removeClass("hidden")
        $evidence.off('load')
      });
    });

    $evidence.hover(
      function() {
        index = $(this).attr('index');
        match = registry[index].detection.match;
        $(this).attr('src', match.dirty);
      },
      function() {
        index = $(this).attr('index');
        match = registry[index].detection.match;
        $(this).attr('src', match.clean);
      }
    );

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

    registry[index].detection.species = response.species_nice
    registry[index].detection.viewpoint = response.viewpoint_nice

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

function drawDetection(response, index) {
    response = response[0]

    registry[index].detection.image = {
      "width": response[0],
      "height": response[1],
    }

    xtl    = 100.0 * registry[index].detection.xtl    / registry[index].detection.image.width
    ytl    = 100.0 * registry[index].detection.ytl    / registry[index].detection.image.height
    width  = 100.0 * registry[index].detection.width  / registry[index].detection.image.width
    height = 100.0 * registry[index].detection.height / registry[index].detection.image.height

    $bbox = $('#image-bbox-' + index)
    $bbox.css({
      'top':    ytl    + '%',
      'left':   xtl    + '%',
      'width':  width  + '%',
      'height': height + '%',
    })

    $bbox.removeClass('image-bbox-hidden')
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

    url = '/api/image/size/?gid_list=[' + registry[index].gid + ']'
    getValue(url, drawDetection, index)

    url = '/api/annot/src/' + bestDetection.id
    $('img#annot-' + index).attr('src', url).on('load', function() {

      $('#annot-label-' + index).show()

      resized()

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

              $('#image-label-' + index).html('Upload: ' + registry[index].file.name)
              $('#image-label-' + index).show()

              resized()

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

function addStats(response, index) {

    $('#stats-container-labels').html(
      '' + '<br/>' +
      'Images:<br/>' +
      'Sightings:<br/>' +
      'Animals:<br/>'
    );

    $('#stats-container-values').html('' +
      '<b>Reference DB</b><br/>' +
      response.images + '<br/>' +
      response.annotations + '<br/>' +
      response.names + '<br/>'
    );

    $("#stats-container").delay(700).fadeIn(1200);
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

      url = '/api/core/db/numbers/'
      getValue(url, addStats, null)

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

        var left    = $('<div class="col-lg-1 col-md-1 col-sm-1 col-xs-1"></div>')
        var element = $('<div class="col-lg-10 col-md-10 col-sm-10 col-xs-10 element" id="element-' + index + '"></div>')
        var right   = $('<div class="col-lg-1 col-md-1 col-sm-1 col-xs-1"></div>')

        row.append(left);
        row.append(element);
        row.append(right);

        var row2 = $('<div class="row" id="row-' + index + '"></div>')
        element.append(row2)

        var left2   = $('<div class="col-lg-4 col-md-4 col-sm-6 col-xs-6 element-left"></div>')
        var center2 = $('<div class="col-lg-4 col-md-4 col-sm-6 col-xs-6 element-center"></div>')
        var right2  = $('<div class="col-lg-4 col-md-4 col-sm-12 col-xs-12 element-right"></div>')

        var evidence = $('<div class="col-lg-12 col-md-12 col-sm-12 col-xs-12"></div>')

        evidence.append(
          '<div class="collapse" id="collapse-' + index + '">' +
          '  <div class="well" style="max-width: 1000px; margin-left: auto; margin-right: auto;">' +
          '    <div class="element-evidence-image-container">' +
          '      <img class="evidence" id="evidence-' + index + '" index="' + index + '" src="">' +
          '      <div class="evidence-label evidence-label-left">Uploaded Query Animal</div>' +
          '      <div class="evidence-label evidence-label-right">Best Matched Reference Animal</div>' +
          '    </div>' +
          '    <div style="width: 100%; text-align: center;"><i style="color: #ccc; margin-top: 20px;">hover to see matched areas</i></div>' +
          '  </div>' +
          '</div>'
        )

        row2.append(left2)
        row2.append(center2)
        row2.append(right2)
        row2.append(evidence);

        var left3 = $('<div class="element-left-image-container"></div>');
        left2.append(left3)

        var image = $('<img class="demo-image" id="image-' + index + '" src="">');

        left3.append(image)
        left3.append($('<div class="image-label" id="image-label-' + index + '"></div>'));
        left3.append($('<div class="image-bbox image-bbox-hidden" id="image-bbox-' + index + '">'));

        var center3 = $('<div class="element-center-image-container"></div>');
        center2.append(center3)

        var annot = $('<img class="demo-image" id="annot-' + index + '" src="">');

        center3.append(annot)
        center3.append($('<div class="annot-label" id="annot-label-' + index + '">Most Confident Detection</div>'));

        var right3 = $('<div class="col-lg-12 col-md-12 col-sm-6 col-xs-6"></div>')

        var texts = ['Upload', 'Detect', 'Classify', 'Identify']
        for (var i2 = 0; i2 < texts.length; i2++) {
          right3.append(
            '<div id="row-container-' + i2 + '" class="row">' +
              '<div class="col-lg-3 col-md-3 col-sm-6 col-xs-6 text-container">' + texts[i2] + '</div>' +
              '<div class="col-lg-9 col-md-9 col-sm-6 col-xs-6 progress-container">' +
                '<div id="progress" class="progress">' +
                  '<div id="progress-bar-' + index + '-' + i2 + '" class="progress-bar progress-bar' + i2 + ' progress-bar-striped active" role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100" style="width: 0%"></div>' +
                '</div>' +
              '</div>' +
            '</div>'
          )
        }

        right2.append(right3)

        right2.append(
          '<div class="col-lg-12 col-md-12 col-sm-6 col-xs-6">' +
            '<div id="id-container-labels-' + index + '" class="col-lg-6 col-md-6 col-sm-6 col-xs-6 id-container-labels"></div>' +
            '<div id="id-container-values-' + index + '" class="col-lg-6 col-md-6 col-sm-6 col-xs-6 id-container-values"></div>' +
          '</div>'
        )

        $('#image-label-' + index).hide()
        $('#annot-label-' + index).hide()

        // $(image).hover(
        //   function() {
        //     if( ! $(this).hasClass('unloaded')) {
        //       $(this).next().show()
        //     }
        //   },
        //   function() {
        //     $(this).next().hide()
        //   }
        // );

        // $(annot).hover(
        //   function() {
        //     if( ! $(this).hasClass('unloaded')) {
        //       $(this).next().show()
        //     }
        //   },
        //   function() {
        //     $(this).next().hide()
        //   }
        // );

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
    $("#logo-container").hide()
    $("#stats-container").hide()
    $('#download-container').hide()

    $('#dropbox-container').removeClass("hidden")
    $('#dropbox-container').css({'height': window.innerHeight})

    offset = (window.innerHeight - $("#dropbox").height()) * 0.5 - 20
    $('#dropbox-container-inside').css('margin-top', offset + "px")
    $("#dropbox-container-inside").hide().fadeIn(1200)

    animations.dropbox = setTimeout(function() {
      $("#dropbox-text").fadeIn(1200)
      $("#logo-container").fadeIn(1200)
      $("#download-container").fadeIn(1200)
    }, 1000);
}

function positionLogoUpload() {

  if (animations.dropbox !== undefined) {
    clearTimeout(animations.dropbox)
  }

  $('#dropbox-text').css({
    'opacity': '0.0',
  })

  $('#dropbox-text').delay(300).css({
    'margin-top': '-150px',
    'margin-bottom': '113px',
    'font-size': '12px',
  })

  $('#dropbox-container-inside').animate({
    'margin-top': "30px",
    'margin-bottom': "10px",
  }, 700);

  $('#dropbox').animate({
    'max-width': "100px",
    'padding': '4px',
  }, 700);

  $('#dropbox-text').delay(500).animate({
    'opacity': '1.0',
  }, 500)

  $('#progress').delay(300).fadeIn(1000)

  $('#download-container').fadeOut(1000)
}

function resized() {
    $('.demo-image').each(function(index) {
        $(this).next().css('left', this.offsetLeft);
    })

    $('.demo-image').each(function(index) {
        $(this).next().css('left', this.offsetLeft);
    })
}

function scrolled() {
    var offset = window.pageYOffset
    // $("#dropbox-header").css('top', offset)

    if (offset == 0) {
      $("#dropbox-header").removeClass('dropbox-header-shaddow')
    } else {
      $("#dropbox-header").addClass('dropbox-header-shaddow')
    }
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

    $(window).on('resize', function(event) {
        resized()
    })

    $(window).on('scroll', function(event) {
        scrolled()
    })

    function stopDefault(event) {
        event.stopPropagation();
        event.preventDefault();
    }

    $(document).on("dragenter", stopDefault);
    $(document).on("dragover",  stopDefault);
    $(document).on("drop",      stopDefault);
}
