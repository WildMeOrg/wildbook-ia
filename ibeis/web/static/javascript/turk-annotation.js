function update_label()
{
  var viewpoint_strs = [];
  viewpoint_strs[0] = 'Left';
  viewpoint_strs[45] = 'Front-Left';
  viewpoint_strs[90] = 'Front';
  viewpoint_strs[135] = 'Front-Right';
  viewpoint_strs[180] = 'Right';
  viewpoint_strs[225] = 'Back-Right';
  viewpoint_strs[270] = 'Back';
  viewpoint_strs[315] = 'Back-Left';

  value = parseFloat( $("#ia-annotation-viewpoint").val() );
  radians_sub = (value / 360.0) * 2.0;
  $("#ia-viewpoint-label-degrees").html(value);
  $("#ia-viewpoint-label-radians1").html((radians_sub * Math.PI).toFixed(4));
  $("#ia-viewpoint-label-radians2").html((radians_sub).toFixed(4));
  value = Math.round(value / 45.0) * 45;
  if(value == 360) value = 0;
  $("#ia-viewpoint-label-text").html(viewpoint_strs[value]);

  var quality_strs = [];
  // quality_strs[5] = 'Excellent';
  // quality_strs[4] = 'Good';
  // quality_strs[3] = 'OK';
  // quality_strs[2] = 'Poor';
  // quality_strs[1] = 'Junk';
  quality_strs[2] = 'Good';
  quality_strs[1] = 'Poor';

  value = parseFloat( $("#ia-annotation-quality").val() );
  $("#ia-quality-label-value").html(value);
  $("#ia-quality-label-text").html(quality_strs[value]);
}

function add_species()
{
  value = $('input[name="species-add"]').val()
  console.log(value);ww

  $('select[name="viewpoint-species"]')
     .append($("<option></option>")
     .attr("value", value)
     .text(value));
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
      $('input#ia-turk-submit-accept').click();
    }
    else if(key == 32)
    {
      // Space key pressed, submit form as delete
      $('input#ia-turk-submit-delete').click();
    }
    // else if(key == 76)
    // {
    //   // L key pressed, submit form as left
    //   $('input#ia-turk-submit-left').click();
    // }
    // else if(key == 82)
    // {
    //   // R key pressed, submit form as right
    //   $('input#ia-turk-submit-right').click();
    // }
    else if(key == 81)
    {
      // Q key pressed, 1 star
      $("#ia-annotation-quality").val(1);
      update_label();
    }
    else if(key == 87)
    {
      // W key pressed, 2 stars
      $("#ia-annotation-quality").val(2);
      update_label();
    }
    else if(key == 69)
    {
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
    else if(key == 80)
    {
      // P key pressed, follow previous link
      $('a#ia-turk-previous')[0].click();
    }
    else if(49 <= key && key <= 56)
    {
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
      $("#ia-annotation-viewpoint").val(value * 45); // multiply number by 45 degrees
      update_label();
    }
    else if(97 <= key && key <= 104)
    {
      // 48 ==  number pad key 0
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
      $("#ia-annotation-viewpoint").val(value * 45); // multiply number by 45 degrees
      update_label();
    }
  }
});