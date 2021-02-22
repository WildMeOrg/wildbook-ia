function update_label()
{
  var viewpoint_strs = [];
  viewpoint_strs[0] = 'Left';
  viewpoint_strs[1] = 'Front-Left';
  viewpoint_strs[2] = 'Front';
  viewpoint_strs[3] = 'Front-Right';
  viewpoint_strs[4] = 'Right';
  viewpoint_strs[5] = 'Back-Right';
  viewpoint_strs[6] = 'Back';
  viewpoint_strs[7] = 'Back-Left';

  value = parseFloat( $("#ia-annotation-viewpoint").val() );
  $("#ia-viewpoint-label-text").html(viewpoint_strs[value]);
}

function add_species()
{
  value = $('input[name="species-add"]').val()
  console.log(value);

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
      $('input#review-submit-accept').click();
    }
    else if(key == 32)
    {
      // Space key pressed, submit form as delete
      $('input#review-submit-delete').click();
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
      $("#slider-viewpoint").val(value);
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
      $("#slider-viewpoint").val(value);
      update_label();
    }
  }
});
