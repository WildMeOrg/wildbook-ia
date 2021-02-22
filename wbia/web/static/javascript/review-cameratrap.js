var hotkeys_disabled = false;

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
    else if(key == 80)
    {
      // P key pressed, follow previous link
      $('a#review-previous')[0].click();
    } else if (key == 84) {
        // T pressed
        $('#ia-cameratrap-toggle').bootstrapToggle('toggle');
    }
  }
});
