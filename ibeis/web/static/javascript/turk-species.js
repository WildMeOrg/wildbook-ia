function add_species() {
  value = $('input[name="species-add"]').val()
  console.log(value);

  $('select[name="ia-species-species"]')
    .append($("<option></option>")
      .attr("value", value)
      .text(value));
  $('select[name="ia-species-species"] option[value="' + value + '"]').prop('selected', true)
}

$(window).keydown(function(event) {
  key = event.which;
  console.log(key);
  console.log('disabled ' + hotkeys_global_disabled);

  if (!hotkeys_global_disabled) {
    if (key == 13) {
      // Enter key pressed, submit form as accept
      $('input#ia-turk-submit-accept').click();
    } else if (key == 32) {
      // Space key pressed, submit form as delete
      $('input#ia-turk-submit-delete').click();
    }
    else if(key == 76)
    {
      // L key pressed, submit form as left
      $('input#ia-turk-submit-left').click();
    }
    else if(key == 82)
    {
      // R key pressed, submit form as right
      $('input#ia-turk-submit-right').click();
    }
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
    } else if (key == 80) {
      // P key pressed, follow previous link
      $('a#ia-turk-previous')[0].click();
    }
  }
});