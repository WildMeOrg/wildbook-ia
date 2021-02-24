
function is_biologist()
{
  console.log('biologist');
  $('#biologist-approval').html('<h3>Thank you, please continue</h3>');
  return true
}

function update_label()
{
  var demographics_sex_strs = [];
  demographics_sex_strs[4] = 'Indeterminate';
  demographics_sex_strs[3] = 'Male';
  demographics_sex_strs[2] = 'Female';
  demographics_sex_strs[1] = 'Unknown';

  var demographics_age_strs = [];
  demographics_age_strs[7] = '36+ Months';
  demographics_age_strs[6] = '24-36 Months';
  demographics_age_strs[5] = '12-24 Months';
  demographics_age_strs[4] = '6-12 Months';
  demographics_age_strs[3] = '3-6 Months';
  demographics_age_strs[2] = '0-3 Months';
  demographics_age_strs[1] = 'Unknown';

  value_sex = parseFloat( $("#slider-demographics-sex").val() );
  value_age = parseFloat( $("#slider-demographics-age").val() );
  $("#label-sex").html(demographics_sex_strs[value_sex]);
  $("#label-age").html(demographics_age_strs[value_age]);
}

$(window).keydown(function(event) {
  key = event.which;
  console.log(key);
  if(key == 13)
  {
    // Enter key pressed, submit form as accept
    $('input#review-submit-accept').click();
  }
  else if(key == 32)
  {
    // Space key pressed, submit form as delete
    $('input#review-submit-clear').click();
  }
  else if(key == 85 || key == 56 || key == 56 || key == 104)
  {
    // 'U' & 8 for 'Unknown'
    $("#slider-demographics-sex").val(1);
    update_label();
  }
  else if(key == 70 || key == 57 || key == 57 || key == 105)
  {
    // 'F' & 9 for 'Female'
    $("#slider-demographics-sex").val(2);
    update_label();
  }
  else if(key == 77 || key == 48 || key == 48 || key == 48)
  {
    // 'M' & 0 for 'Male'
    $("#slider-demographics-sex").val(3);
    update_label();
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
    $("#slider-demographics-age").val(value + 1); // subtract 2 to get into the range [-1, -5]
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
    $("#slider-demographics-age").val(value + 1); // subtract 2 to get into the range [-1, -5]
    update_label();
  }
  else if(key == 90)
  {
    // Z key pressed, zoom
    margin = 5

    if (zoom) {
      $('#big-image').removeClass('big-image-zoom')
      scroll = 0
    } else {
      $('#big-image').addClass('big-image-zoom')
      offset = $('#big-image').offset()
      scroll = offset.top - margin
    }

    if (animation != null) {
      animation.stop();
    }

    animation = $('html, body').animate({scrollTop: scroll});

    zoom = ! zoom
    fix_height()
  }
});
