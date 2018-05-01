function check_form() {
    // Resave the current entries to the form
    data = cta.resize();

    if (data != null) {
        return true
    }

    response = confirm('You must specify a contour for each image.  Are you sure you want to continue with NO contour?')
    return response
}

function update_radius() {
    radius = $('#slider-radius').val()
    radius = parseInt(radius)
    cta.update_radius(radius)
}

function update_radius_slider(delta) {
    slider = $('#slider-radius')
    radius = slider.val()
    radius = parseInt(radius)
    radius += delta
    radius = slider.val(radius)

    update_radius()
}

$(window).keydown(function(event) {
    key = event.which;

    if( ! hotkeys_global_disabled) {
        console.log(key)

        if (key == 13) {
            // Enter key pressed
            $('input#ia-turk-submit-accept').click();
        } else if (key == 81) {
            // Q key pressed
            update_radius_slider(1)
        } else if (key == 65) {
            // A key pressed
            update_radius_slider(-1)
        } else if (key == 82) {
            // R key pressed
            cta.reset_interaction()
        } else if (key == 80) {
            // P key pressed, follow previous link
            $('a#ia-turk-previous')[0].click();
        }
    }
});