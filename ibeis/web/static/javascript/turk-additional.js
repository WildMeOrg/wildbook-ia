
function is_biologist()
{
  console.log('biologist');
  $('#biologist-approval').html('<h3>Thank you, please continue</h3>');
  return true
}

function update_label()
{
  var additional_sex_strs = [];
  additional_sex_strs[3] = 'Male';
  additional_sex_strs[2] = 'Female';
  additional_sex_strs[1] = 'Unknown';

  var additional_age_strs = [];
  additional_age_strs[7] = '36+ Months';
  additional_age_strs[6] = '24-36 Months';
  additional_age_strs[5] = '12-24 Months';
  additional_age_strs[4] = '6-12 Months';
  additional_age_strs[3] = '3-6 Months';
  additional_age_strs[2] = '0-3 Months';
  additional_age_strs[1] = 'Unknown';

  value_sex = parseFloat( $("#slider-additional-sex").val() );
  value_age = parseFloat( $("#slider-additional-age").val() );
  $("#label-sex").html(additional_sex_strs[value_sex]);
  $("#label-age").html(additional_age_strs[value_age]);
}

$(window).keydown(function(event) {
  key = event.which;
  console.log(key);
  if(key == 13)
  {
    // Enter key pressed, submit form as accept
    $('input#turk-submit-accept').click();
  }
  else if(key == 32)
  {
    // Space key pressed, submit form as delete
    $('input#turk-submit-clear').click();
  }
  else if(key == 85 || key == 56 || key == 56 || key == 104)
  {
    // 'U' & 8 for 'Unknown'
    $("#slider-additional-sex").val(1);
    update_label();
  }
  else if(key == 70 || key == 57 || key == 57 || key == 105)
  {
    // 'F' & 9 for 'Female'
    $("#slider-additional-sex").val(2);
    update_label();
  }
  else if(key == 77 || key == 48 || key == 48 || key == 48)
  {
    // 'M' & 0 for 'Male'
    $("#slider-additional-sex").val(3);
    update_label();
  }
  else if(key == 80)
  {
    // P key pressed, follow previous link
    $('a#turk-previous')[0].click();
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
    $("#slider-additional-age").val(value + 1); // subtract 2 to get into the range [-1, -5]
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
    $("#slider-additional-age").val(value + 1); // subtract 2 to get into the range [-1, -5]
    update_label();
  }
});