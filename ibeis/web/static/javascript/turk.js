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
//   console.log(key);
  if(key == 13)
  {
    // Enter key pressed, submit form as accept
    $('input#viewpoint-submit-accept').click();
  }
  else if(key == 32)
  {
    // Space key pressed, submit form as skip
    $('input#viewpoint-submit-skip').click();
  }
  else if(49 <= key && key <= 56)
  {
    // 48 == numberc key 0
    // 49 == numeric key 1
    // 50 == numberc key 2
    // 51 == numberc key 3
    // 52 == numberc key 4
    // 53 == numberc key 5
    // 54 == numberc key 6
    // 55 == numberc key 7
    // 56 == numberc key 8
    // 57 == numberc key 9
    value = key - 49; // offset by 49 so that the number one is the value of 0
    $("#slider-viewpoint").val(value * 45); // multiply number by 45 degrees 
    update_label();
  }
});