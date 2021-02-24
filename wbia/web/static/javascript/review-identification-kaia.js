
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

  hotkeys_disabled = true;

  if( ! hotkeys_disabled)
  {
    if(key == 84 || key == 49 || key == 97)
    {
      // T key pressed + 1
      $('input#ia-review-submit-match').click();
    }
    else if(key == 70 || key == 50 || key == 98)
    {
      // F key pressed + 2
      $('input#ia-review-submit-notmatch').click();
    }
    else if(key == 67 || key == 51 || key == 99)
    {
      // C key pressed + 3
      // $('input#ia-review-submit-notcomparable').click();
    }
    else if(key == 81)
    {
      // Q key pressed
      $('#ia-review-photobomb').trigger('click');
    }
    else if(key == 87)
    {
      // W key pressed
      $('#ia-review-scenerymatch').trigger('click');
    }
    else if(key == 80)
    {
      // P key pressed
      $('a#ia-review-previous')[0].click();
    }
    else if(key == 90)
    {
      // Z key pressed, zoom
      margin = 5

      if (zoom) {
        $('#identification-image-clean').removeClass('identification-image-zoom')
        $('#identification-image-matches').removeClass('identification-image-zoom')
        scroll = 0
      } else {
        $('#identification-image-clean').addClass('identification-image-zoom')
        $('#identification-image-matches').addClass('identification-image-zoom')
        offset = $('#identification-image-clean').offset()
        scroll = offset.top - margin
      }

      if (animation != null) {
        animation.stop();
      }

      animation = $('html, body').animate({scrollTop: scroll});

      zoom = ! zoom
      fix_height()
    }
  }
});
