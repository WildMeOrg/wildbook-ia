function update_label() {
  var viewpoint_strs = [];
  viewpoint_strs[-1] = 'Unspecified';
  viewpoint_strs[0] = 'Left';
  viewpoint_strs[1] = 'Front-Left';
  viewpoint_strs[2] = 'Front';
  viewpoint_strs[3] = 'Front-Right';
  viewpoint_strs[4] = 'Right';
  viewpoint_strs[5] = 'Back-Right';
  viewpoint_strs[6] = 'Back';
  viewpoint_strs[7] = 'Back-Left';

  value = parseFloat($("#ia-annotation-viewpoint").val());
  $("#ia-annotation-viewpoint-label-text").html(viewpoint_strs[value]);

  var quality_strs = [];
  quality_strs[-1] = 'Unspecified';
  quality_strs[0] = 'Poor';
  quality_strs[1] = 'Good';

  value = parseFloat($("#ia-annotation-quality").val());
  $("#ia-quality-label-text").html(quality_strs[value]);
}

function add_species() {
  value = $('input[name="species-add"]').val()
  console.log(value);

  $('select[name="ia-annotation-species"]')
    .append($("<option></option>")
      .attr("value", value)
      .text(value));
  $('select[name="ia-annotation-species"] option[value="' + value + '"]').prop('selected', true)
}

$(window).keydown(function(event) {
  key = event.which;
  console.log(key);
  console.log('disabled ' + hotkeys_global_disabled);

  if (!hotkeys_global_disabled) {
    if (key == 13) {
      // Enter key pressed, submit form as accept
      $('input#ia-review-submit-accept').click();
    } else if (key == 32) {
      // Space key pressed, submit form as delete
      $('input#ia-review-submit-delete').click();
    }
    else if(key == 76)
    {
      // L key pressed, submit form as left
      $('input#ia-review-submit-left').click();
    }
    else if(key == 82)
    {
      // R key pressed, submit form as right
      $('input#ia-review-submit-right').click();
    }
    else if (key == 69) {
      // E key pressed, No quality
      $("#ia-annotation-quality").val(-1);
      update_label();
    } else if (key == 81) {
      // Q key pressed, 1 star
      $("#ia-annotation-quality").val(0);
      update_label();
    } else if (key == 87) {
      // W key pressed, 2 stars
      $("#ia-annotation-quality").val(1);
      update_label();
    } else if (key == 79) {
      $('#ia-annotation-multiple').trigger('click');
    }
    // else if(key == 69)
    // {
    //   $('#ia-annotation-multiple').trigger('click');
    //   E key pressed, 3 starts
    //   $("#ia-annotation-quality").val(3);
    //   update_label();
    // }
    // else if(key == 82)
    // {
    //   // R key pressed, 4 stars
    //   $("#ia-annotation-quality").val(4);
    //   update_label();
    // }
    // else if(key == 84)
    // {
    //   // T key pressed, 5 stars
    //   $("#ia-annotation-quality").val(5);
    //   update_label();
    // }
    // else if(key == 85)
    // {
    //   // U key pressed, select Unspecified Animal in selection box
    //   $('select[name="viewpoint-species"]').val("unspecified_animal");
    // }
    else if (key == 67) {
      // C pressed
      var element = $("#ia-annotation-species")
      var children = element.children('option')
      var length = children.length;
      var index = element.find("option:selected").index()
        // Increment
      if (event.shiftKey) {
        index -= 1
      } else {
        index += 1
      }
      index = index % length
      children.eq(index).prop('selected', true);
    } else if (key == 80) {
      // P key pressed, follow previous link
      $('a#ia-review-previous')[0].click();
    } else if (48 <= key && key <= 56) {
      // 48 == numeric key 0
      // 49 == numeric key 1
      // 50 == numeric key 2
      // 51 == numeric key 3
      // 52 == numeric key 4
      // 53 == numeric key 5
      // 54 == numeric key 6
      // 55 == numeric key 7
      // 56 == numeric key 8
      // 57 == numeric key 9
      value = key - 49; // offset by 49 so that the number one is the value of 0
      $("#ia-annotation-viewpoint").val(value);
      update_label();
    } else if (96 <= key && key <= 104) {
      // 96 ==  number pad key 0
      // 97 ==  number pad key 1
      // 98 ==  number pad key 2
      // 99 ==  number pad key 3
      // 100 == number pad key 4
      // 101 == number pad key 5
      // 102 == number pad key 6
      // 103 == number pad key 7
      // 104 == number pad key 8
      // 105 == number pad key 9
      value = key - 97; // offset by 97 so that the number one is the value of 0
      $("#ia-annotation-viewpoint").val(value);
      update_label();
    }
  }
});
