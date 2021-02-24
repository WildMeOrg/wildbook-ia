function hide_viewpoint_2_axis_panel(index) {
    $(".ia-viewpoint-axis-" + index).hide()
    if(index == 3) {
        $("#ia-viewpoint-axis-2-hotkey").css('opacity', 0.0)
    }
}

function update_label() {
    var viewpoint_strs = [];
    viewpoint_strs[-1] = null;
    viewpoint_strs[0] = 'Top';
    viewpoint_strs[1] = 'Bottom';
    viewpoint_strs[2] = 'Front';
    viewpoint_strs[3] = 'Back';
    viewpoint_strs[4] = 'Left';
    viewpoint_strs[5] = 'Right';

    for (var index = 0; index <= 6; index++) {
        $("#col-viewpoint-ticks-2-" + index).css({
            opacity: 1.0,
        })
        $("#col-viewpoint-ticks-3-" + index).css({
            opacity: 1.0,
        })
    }

    value1 = parseFloat($("#ia-detection-annotation-viewpoint-1").val());
    value2 = parseFloat($("#ia-detection-annotation-viewpoint-2").val());
    value3 = parseFloat($("#ia-detection-annotation-viewpoint-3").val());

    if (value3 >= 0 && value1 == -1) {
        value3 = -1
        $("#ia-detection-annotation-viewpoint-3").val(value3)
    }
    if (value3 >= 0 && value2 == -1) {
        value3 = -1
        $("#ia-detection-annotation-viewpoint-3").val(value3)
    }
    if (value2 >= 0 && value1 == -1) {
        value2 = -1
        $("#ia-detection-annotation-viewpoint-2").val(value2)
    }

    if (value1 >= 0) {
        invalid1 = 2 * Math.floor(value1 / 2.0)
        invalid2 = invalid1 + 1
        $("#col-viewpoint-ticks-2-" + (invalid1 + 1)).css({
            opacity: 0.0,
        })
        $("#col-viewpoint-ticks-2-" + (invalid2 + 1)).css({
            opacity: 0.0,
        })
        $("#col-viewpoint-ticks-3-" + (invalid1 + 1)).css({
            opacity: 0.0,
        })
        $("#col-viewpoint-ticks-3-" + (invalid2 + 1)).css({
            opacity: 0.0,
        })

        if (value2 == invalid1 || value2 == invalid2) {
            value2 = -1
            value3 = -1
            $("#ia-detection-annotation-viewpoint-2").val(value2)
            $("#ia-detection-annotation-viewpoint-3").val(value2)
        }

        if (value2 >= 0) {
            invalid3 = 2 * Math.floor(value2 / 2.0)
            invalid4 = invalid3 + 1
            $("#col-viewpoint-ticks-3-" + (invalid3 + 1)).css({
                opacity: 0.0,
            })
            $("#col-viewpoint-ticks-3-" + (invalid4 + 1)).css({
                opacity: 0.0,
            })

            if (value3 == invalid1 || value3 == invalid2 || value3 == invalid3 || value3 == invalid4) {
                value3 = -1
                $("#ia-detection-annotation-viewpoint-3").val(value3)
            }
        }
    }

    tag1 = viewpoint_strs[value1]
    tag2 = viewpoint_strs[value2]
    tag3 = viewpoint_strs[value3]

    if (tag1 != null) {
        if (tag2 != null) {
            if (tag3 != null) {
                tag = tag1 + "-" + tag2 + "-" + tag3
            } else {
                tag = tag1 + "-" + tag2
            }
        } else {
            tag = tag1
        }
    } else {
        tag = "<i>Unspecified</i>"
    }

    $("#ia-detection-annotation-viewpoint-label").html(tag);

    var quality_strs = [];
    quality_strs[1] = 'Poor';
    quality_strs[2] = 'Good';

    value = parseFloat($("#ia-detection-annotation-quality").val());
    $("#ia-detection-annotation-quality-label").html(quality_strs[value]);
}

function add_species()
{
  value = $('input[name="species-add"]').val()
  console.log(value);

  $('select[name="viewpoint-species"]')
     .append($("<option></option>")
     .attr("value", value)
     .text(value));
  $('select[name="viewpoint-species"] option[value="' + value + '"]').prop('selected', true)
}

var hotkeys_disabled = false;

$('#species-add').on('shown.bs.modal', function () {
  $('input[name="species-add"]').val('')
  hotkeys_disabled = true;
});

$('#species-add').on('hidden.bs.modal', function () {
  hotkeys_disabled = false;
});


$(window).keydown(function(event) {
  key = event.which;
  console.log(key);
  console.log('disabled ' + hotkeys_disabled);

  if( ! hotkeys_disabled)
  {
    if(key == 13)
    {
      // Enter key pressed, submit form as accept
      $('input#review-submit-accept').click();
    }
    else if (key == 17) {
      // Ctrl pressed
      $('.ia-detection-hotkey').show();
    }
    else if(key == 32)
    {
      // Space key pressed, submit form as delete
      $('input#review-submit-delete').click();
    }
    else if (key == 67) {
      // C pressed
      var element = $('select[name="viewpoint-species"]')
      var children = element.children('option')
      var length = children.length;
      var index = element.find("option:selected").index()
      // Increment
      if(event.shiftKey) {
          index -= 1
      } else {
          index += 1
      }
      index = index % length
      children.eq(index).prop('selected', true);
    }
    else if(key == 76)
    {
      // L key pressed, submit form as left
      $('input#review-submit-left').click();
    }
    else if(key == 82)
    {
      // R key pressed, submit form as right
      $('input#review-submit-right').click();
    }
    else if(key == 85)
    {
      // U key pressed, select Unspecified Animal in selection box
      $('select[name="viewpoint-species"]').val("unspecified_animal");
    }
    else if(key == 90)
    {
      // Z key pressed, select Plains Zebra in selection box
      $('select[name="viewpoint-species"]').val("zebra_plains");
    }
    else if(key == 80)
    {
      // P key pressed, follow previous link
      $('a#review-previous')[0].click();
    }
    else if ((48 <= key && key <= 55) || (96 <= key && key <= 103)) {
        // 48 == 96  == numeric key 0
        // 49 == 97  == numeric key 1
        // 50 == 98  == numeric key 2
        // 51 == 99  == numeric key 3
        // 52 == 100 == numeric key 4
        // 53 == 101 == numeric key 5
        // 54 == 102 == numeric key 6
        // 55 == 103 == numeric key 7
        // 56 == 104 == numeric key 8
        // 57 == 105 == numeric key 9

        value1 = parseFloat($("#ia-detection-annotation-viewpoint-1").val());
        value2 = parseFloat($("#ia-detection-annotation-viewpoint-2").val());

        if (value1 == -1 && event.altKey) {
            return
        }

        if (!axis2 || value1 == -1 || event.shiftKey) {
            element = "#ia-detection-annotation-viewpoint-1"
        } else if (!axis3 || value2 == -1 || event.altKey) {
            element = "#ia-detection-annotation-viewpoint-2"
        } else {
            element = "#ia-detection-annotation-viewpoint-3"
        }

        if (key == 48 || key == 96) {
            value = -1
        } else if (49 <= key && key <= 56) {
            value = key - 49;
        } else {
            value = key - 97;
        }

        $(element).val(value).trigger('change');
    }
  }
});
