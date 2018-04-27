function check_form() {
    // Resave the current entries to the form
    cta.resize();

    if (cta.entries.length > 0) {
        return true
    }

    response = confirm('You must specify a contour for each image.  Are you sure you want to continue with NO contour?')
    return response
}

$(window).keydown(function(event) {
    key = event.which;

    if( ! hotkeys_global_disabled) {
        console.log(key)

        if (key == 17) {
            // Ctrl pressed
            $('.ia-detection-hotkey').show();
        } else if (key == 13) {
            // Enter key pressed
            $('input#ia-turk-submit-accept').click();
        }
        else if (key == 80) {
            // P key pressed, follow previous link
            $('a#ia-turk-previous')[0].click();
        }
    }
});

$(window).keyup(function(event) {
    key = event.which;

    if( ! hotkeys_global_disabled) {
        if (key == 17) {
            // Ctrl pressed
            $('.ia-detection-hotkey').hide();
        }
    }
});