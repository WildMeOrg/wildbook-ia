
function update_label()
{
  var quality_strs = [];
  quality_strs[5] = 'Excellent';
  quality_strs[4] = 'Good';
  quality_strs[3] = 'OK';
  quality_strs[2] = 'Poor';
  quality_strs[1] = 'Junk';

  value = parseFloat( $("#slider-quality").val() );
  $("#label-value").html(value);
  $("#label-text").html(quality_strs[value]);
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
    $('input#review-submit-delete').click();
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
    $("#slider-quality").val(value + 1); // subtract 2 to get into the range [-1, -5]
    update_label();
  }
  else if(97 <= key && key <= 104)
  {
    // 48 ==  number pad key 0
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
    $("#slider-quality").val(value + 1); // subtract 2 to get into the range [-1, -5]
    update_label();
  }
});
