function check_form(clear) {
  update_buttons();
  return true;
}

function add_species() {
  value = $('input[name="species-add"]').val()

  $('select[name="ia-species-species"]')
    .append($("<option></option>")
      .attr("value", value)
      .text(value));
  $('select[name="ia-species-species"] option[value="' + value + '"]').prop('selected', true)

  update_buttons();
}

function update_buttons(clicked) {
  selected = $('.ia-species-button[selected="selected"]')

  if (clicked === undefined) {
    clicked = selected
  } else {
    if (clicked.attr('id') == selected.attr('id')) {
      clicked = undefined
    }
  }

  $('.ia-species-button').attr('selected', false)
  $('.ia-species-button').removeClass('btn-success')
  $('.ia-species-button').addClass('btn-default')

  if (clicked !== undefined) {
    val = clicked.attr('value');
    $(clicked).attr('selected', true)
    $(clicked).removeClass('btn-default')
    $(clicked).addClass('btn-success')
  } else {
    val = undefined
  }

  other_select = $('#ia-species-select-other')
  if (val == "____") {
    other_select.show()
    val = $("#ia-species-select-other :selected").val();
  } else {
    other_select.hide()
  }

  $('input[name="ia-species-value"]').val(val)
}

$(window).keydown(function(event) {
  key = event.which;
  console.log(key);
  console.log('disabled ' + hotkeys_global_disabled);

  if (!hotkeys_global_disabled) {
    if (key == 13) {
      // Enter key pressed, submit form as accept
      $('input#ia-review-submit-accept').click();
    }
    else if (49 <= key && key <= 57) {
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
      // Number key pressed, submit form as delete
      key -= 48
      update_buttons($('#ia-species-button-' + key))
    }
    else if (97 <= key && key <= 105) {
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
      // Number key pressed, submit form as delete
      key -= 96
      update_buttons($('#ia-species-button-' + key))
    }
    else if (key == 32) {
      // Space key pressed, submit form as delete
      $('input#ia-review-submit-skip').click();
      event.preventDefault()
    }
    // else if(key == 76)
    // {
    //   // L key pressed, submit form as left
    //   $('input#ia-review-submit-left').click();
    // }
    // else if(key == 82)
    // {
    //   // R key pressed, submit form as right
    //   $('input#ia-review-submit-right').click();
    // }
    else if (key == 67) {
      // C pressed
      var element = $("#ia-species-species")
      var children = element.children('option')
      var length = children.length;
      var index = element.find("option:selected").index()
        // Increment
      if (event.shiftKey) {
        index -= 1
      } else {
        index += 1
      }
      index = index % length
      children.eq(index).prop('selected', true);
      update_buttons()
    } else if (key == 80) {
      // P key pressed, follow previous link
      $('a#ia-review-previous')[0].click();
    }
  }
});
