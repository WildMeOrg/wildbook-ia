function update_label()
{
  value = parseFloat( $("#slider-viewpoint").val() );
  radians_sub = (value / 360.0) * 2.0;
  $("#label-degrees").html(value);
  $("#label-radians1").html((radians_sub * Math.PI).toFixed(4));
  $("#label-radians2").html((radians_sub).toFixed(4));
}

$("body").keydown(function(event) {
  key = event.which;
  console.log(key);
  if(key == 13)
  {
    // Enter key pressed, submit form as accept
    $('input#turk-submit-accept').click();
  }
  else if(key == 32)
  {
    // Space key pressed, submit form as skip
    $('input#turk-submit-skip').click();
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
    $("#slider-viewpoint").val(value * 45); // multiply number by 45 degrees 
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
    $("#slider-viewpoint").val(value * 45); // multiply number by 45 degrees 
    update_label();
  }
});