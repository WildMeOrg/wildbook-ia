
var hotkeys_disabled = false;

$('#species-add').on('shown.bs.modal', function () {
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
    if(key == 84 || key == 49 || key == 97)
    {
      // T key pressed + 1
      $('input#ia-turk-submit-match').click();
    }
    else if(key == 70 || key == 50 || key == 98)
    {
      // F key pressed + 2
      $('input#ia-turk-submit-notmatch').click();
    }
    else if(key == 67 || key == 51 || key == 99)
    {
      // C key pressed + 3
      $('input#ia-turk-submit-notcomparable').click();
    }
    else if(key == 80 || key == 52 || key == 100)
    {
      // P key pressed + 4
      $('input#ia-turk-submit-photobomb').click();
    }
    else if(key == 83 || key == 53 || key == 101)
    {
      // S key pressed + 5
      $('input#ia-turk-submit-scenerymatch').click();
    }
    else if(key == 81)
    {
      // Q key pressed
      $('a#ia-turk-previous')[0].click();
    }
  }
});