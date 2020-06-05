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

        var left    = $('<div class="col-lg-1  col-md-1  col-sm-1  col-xs-1 "></div>')
        var element = $('<div class="col-lg-10 col-md-10 col-sm-10 col-xs-10 element" id="element-' + index + '"></div>')
        var right   = $('<div class="col-lg-1  col-md-1  col-sm-1  col-xs-1 "></div>')

        row.append(left);
        row.append(element);
        row.append(right);

        var row2 = $('<div class="row" id="row-' + index + '"></div>')
        element.append(row2)

        var left2   = $('<div class="col-lg-12 col-md-12 col-sm-12 col-xs-12 element-left upload"></div>')
        var right2  = $('<div class="col-lg-12 col-md-12 col-sm-12 col-xs-12 element-right upload"></div>')

        row2.append(left2)
        row2.append(right2)

        var image = $('<img id="image-' + index + '" src="">');
        left2.append(image)

        var texts = ['Upload']
        for (var i2 = 0; i2 < texts.length; i2++) {
          right2.append(
            '<div id="row-container-' + i2 + '" class="row">' +
              '<div class="col-lg-12 col-md-12 col-sm-12 col-xs-12 progress-container upload">' +
                '<div id="progress" class="progress upload">' +
                  '<div id="progress-bar-' + index + '-' + i2 + '" class="progress-bar progress-bar3 progress-bar-striped active" role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100" style="width: 0%"></div>' +
                '</div>' +
              '</div>' +
            '</div>'
          )
        }

        right2.append(
            '<div id="id-container-labels-' + index + '" class="col-lg-6 col-md-6 col-sm-6 col-xs-6 id-container-labels"></div>' +
            '<div id="id-container-values-' + index + '" class="col-lg-6 col-md-6 col-sm-6 col-xs-6 id-container-values"></div>'
          )

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

    $('#dropbox-container').removeClass("hidden")
    // $('#dropbox-container').css({'height': window.innerHeight})

    offset = (window.innerHeight - $("#dropbox").height()) * 0.5
    // $('#dropbox-container-inside').css('margin-top', offset + "px")
    $("#dropbox-container-inside").hide().fadeIn(1200)

    animations.dropbox = setTimeout(function() {
      $("#dropbox-text").fadeIn(1200)
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

function init(image1) {
    var $dropbox = $("#dropbox");

    $dropbox.attr('src', image1).on('load', function() {
      positionLogoInit();
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
