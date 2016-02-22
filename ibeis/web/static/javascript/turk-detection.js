$(window).keydown(function(event) {
  key = event.which;
  console.log(key)
  if(key == 13)
  {
    // Enter key pressed, submit form as accept
    $('input#ia-turk-submit-accept').click();
  }
  else if(key == 32)
  {
    // Space key pressed, submit form as delete
    $('input#ia-turk-submit-clear').click();
  }
  else if(key == 80)
  {
    // P key pressed, follow previous link
    $('a#ia-turk-previous')[0].click();
  }
});