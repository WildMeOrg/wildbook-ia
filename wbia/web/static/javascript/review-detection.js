function hide_viewpoint_2_axis_panel(index) {
    $(".ia-viewpoint-axis-" + index).hide()
    if(index == 3) {
        $("#ia-viewpoint-axis-2-hotkey").css('opacity', 0.0)
    }
}

function check_form(skipcheck) {
    if (skipcheck) {
        return true;
    }

    // Recalculate current values and update form data field before submitting
    bba.resize();

    if(config_interest_bypass) {
        return true
    }

    if (bba.entries.length == 0) {
        return true
    }

    if (bba.entries.length == 1 && config_autointerest) {
        bba.entries[0].highlighted = true
        bba.refresh()
    }
    for (var index = 0; index < bba.entries.length; index++) {
        if (bba.entries[index].highlighted) {
            return true
        }
    }
    response = confirm('Most images that have at least one annotation should have one labeled as being "of interest".  You have highlighted none.  This is fine only if there really are no clear, good quality annotations.\n\nClick Cancel to return to the image and select an annotation of interest, and click Ok to accept the current results and continue to the next image.')
    return response
}

function update_metadata_panel(state) {
    var css_active, css_inactive

    state !== undefined || (state = null)
    states.active = state

    css_active = {
        "background-color": "#286090",
        "color": "#FFFFFF",
        "border": "1px solid #286090",
    }
    css_inactive = {
        "background-color": "#FFFFFF",
        "color": "#333333",
        "border": "1px solid #333333",
    }


    if (state == "annotation") {
        active_ids = ["#ia-metadata-badge-annotation"]
        inactive_ids = ["#ia-metadata-badge-part"]
        visible_ids = ['#ia-metadata-panel-annotation']
        hidden_ids = ['#ia-metadata-panel-part']
    } else if (state == "part") {
        active_ids = ["#ia-metadata-badge-part"]
        inactive_ids = ["#ia-metadata-badge-annotation"]
        visible_ids = ['#ia-metadata-panel-part']
        hidden_ids = ['#ia-metadata-panel-annotation']
    } else {
        $('#ia-metadata-rowid').text('');
        active_ids = []
        inactive_ids = ["#ia-metadata-badge-annotation", "#ia-metadata-badge-part"]
        visible_ids = []
        hidden_ids = ['#ia-metadata-panel-annotation', '#ia-metadata-panel-part']
    }

    if ( ! config_metadata) {
        hidden_ids.concat(visible_ids)
        visible_ids = []
    }

    if ( (! config_metadata) || (single_mode && ! config_parts) ) {
        $('#ia-detection-setting-wrapper').css('display', 'none');
        $('form.ia-form').css('padding-top', '15px');
    }

    if ( ! config_viewpoint) {
        hidden_ids.push("#ia-metadata-panel-annotation-viewpoint")
        $("#ia-metadata-panel-annotation-row2").css('margin-top', '10px');
        hidden_ids.push("#ia-metadata-panel-part-viewpoint")
    }

    if ( ! config_quality) {
        hidden_ids.push("#ia-metadata-panel-annotation-quality")
        hidden_ids.push("#ia-metadata-panel-part-quality")
    }

    if ( ! config_quality && config_flags) {
        $("#ia-metadata-panel-annotation-species").css('margin-top', '-55px');
        $("#ia-metadata-panel-part-type").css('margin-top', '-26px');
    }

    if ( ! config_flags) {
        hidden_ids.push("#ia-metadata-panel-annotation-flags")
    }

    if ( ! config_flags_aoi) {
        hidden_ids.push("#ia-metadata-panel-annotation-flags-aoi")
    }

    if ( ! config_flags_multiple) {
        hidden_ids.push("#ia-metadata-panel-annotation-flags-multiple")
    }

    if ( ! config_species) {
        hidden_ids.push("#ia-metadata-panel-annotation-species")
        hidden_ids.push("#ia-metadata-panel-part-type")
    }

    if( single_mode ) {
        hidden_ids.push("#ia-detection-setting-toggle-wrapper")
        hidden_ids.push("#ia-detection-setting-orientation-wrapper")
    }

    if ( ! config_label) {
        hidden_ids.push("#ia-metadata-badge-annotation")
        hidden_ids.push("#ia-metadata-badge-part")
    }

    if ( ! config_quickhelp) {
        hidden_ids.push("#ia-metadata-panel-annotation-quickhelp")
        hidden_ids.push("#ia-metadata-panel-annotation-quickhelp-fallback")
    }

    for (var index = 0; index < active_ids.length; index++) {
        $(active_ids[index]).css(css_active)
    }
    for (var index = 0; index < inactive_ids.length; index++) {
        $(inactive_ids[index]).css(css_inactive)
    }
    for (var index = 0; index < visible_ids.length; index++) {
        $(visible_ids[index]).show()
    }
    for (var index = 0; index < hidden_ids.length; index++) {
        $(hidden_ids[index]).hide()
    }

    fix_metadata_panels();
}

function enable_metadata_annotations() {
    $('#ia-detection-annotation-viewpoint-1').prop("disabled", false);
    $('#ia-detection-annotation-viewpoint-2').prop("disabled", false);
    $('#ia-detection-annotation-viewpoint-3').prop("disabled", false);
    $('#ia-detection-annotation-quality').prop("disabled", false);
    $('#ia-detection-annotation-multiple').prop("disabled", false);
    $('#ia-detection-annotation-interest').prop("disabled", false);
    $('#ia-detection-class').prop("disabled", false);
    $('span[data-target="#species-add"]').css("visibility", "visible");
    $("#ia-metadata-panel-annotation").css("color", "#000");
    $("#ia-detection-form-annotation-warning").css("visibility", "hidden");
}

function disable_metadata_annotations() {
    $('#ia-detection-annotation-viewpoint-1').prop("disabled", true);
    $('#ia-detection-annotation-viewpoint-2').prop("disabled", true);
    $('#ia-detection-annotation-viewpoint-3').prop("disabled", true);
    $('#ia-detection-annotation-quality').prop("disabled", true);
    $('#ia-detection-annotation-multiple').prop("disabled", true);
    $('#ia-detection-annotation-interest').prop("disabled", true);
    $('#ia-detection-class').prop("disabled", true);
    $('span[data-target="#species-add"]').css("visibility", "hidden");
    $("#ia-metadata-panel-annotation").css("color", "#777");
    $("#ia-detection-form-annotation-warning").css("visibility", "visible");
}

function enable_metadata_parts() {
    $('#ia-detection-part-viewpoint-1').prop("disabled", false);
    $('#ia-detection-part-quality').prop("disabled", false);
    $('#ia-detection-part').prop("disabled", false);
    $('span[data-target="#part-add"]').css("visibility", "visible");
    $("#ia-metadata-panel-part").css("color", "#000");
    $("#ia-detection-form-part-warning").css("visibility", "hidden");
}

function disable_metadata_parts() {
    $('#ia-detection-part-viewpoint-1').prop("disabled", true);
    $('#ia-detection-part-quality').prop("disabled", true);
    $('#ia-detection-part').prop("disabled", true);
    $('span[data-target="#part-add"]').css("visibility", "hidden");
    $("#ia-metadata-panel-part").css("color", "#777");
    $("#ia-detection-form-part-warning").css("visibility", "visible");
}

function show_annotation_metadata(entry) {
    invalid = false
    entry.metadata.viewpoint1 !== undefined || (invalid = true)
    entry.metadata.viewpoint2 !== undefined || (invalid = true)
    entry.metadata.viewpoint3 !== undefined || (invalid = true)
    entry.metadata.quality    !== undefined || (invalid = true)
    entry.metadata.species    !== undefined || (invalid = true)
    entry.metadata.multiple   !== undefined || (invalid = true)
    entry.highlighted         !== undefined || (invalid = true)
    entry.metadata.viewpoint1 !== undefined || (entry.metadata.viewpoint1 = -1)
    entry.metadata.viewpoint2 !== undefined || (entry.metadata.viewpoint2 = -1)
    entry.metadata.viewpoint3 !== undefined || (entry.metadata.viewpoint3 = -1)
    entry.metadata.quality    !== undefined || (entry.metadata.quality = 0)
    entry.metadata.species    !== undefined || (entry.metadata.species = '____')
    entry.metadata.multiple   !== undefined || (entry.metadata.multiple = false)
    entry.highlighted         !== undefined || (entry.highlighted = false)

    if (entry.label !== null) {
        $('#ia-metadata-rowid').text('Annotation ID: ' + entry.label);
    } else {
        $('#ia-metadata-rowid').text('');
    }
    $('#ia-detection-annotation-viewpoint-1').val(entry.metadata.viewpoint1);
    $('#ia-detection-annotation-viewpoint-2').val(entry.metadata.viewpoint2);
    $('#ia-detection-annotation-viewpoint-3').val(entry.metadata.viewpoint3);
    $('#ia-detection-annotation-quality').val(entry.metadata.quality);
    $('#ia-detection-annotation-class option[value="' + entry.metadata.species + '"]').prop("selected", true);
    $('#ia-detection-annotation-multiple').prop("checked", entry.metadata.multiple);
    $('#ia-detection-annotation-interest').prop("checked", entry.highlighted);

    if(invalid) {
        $('.ia-detection-form-annotation-value').trigger('change');
    }

    update_label();
}

function show_part_metadata(entry, parent_entry) {
    invalid = false
    entry.metadata.viewpoint1     !== undefined || (invalid = true)
    entry.metadata.quality        !== undefined || (invalid = true)
    entry.metadata.part           !== undefined || (invalid = true)
    entry.metadata.viewpoint1     !== undefined || (entry.metadata.viewpoint1 = -1)
    entry.metadata.quality        !== undefined || (entry.metadata.quality = 0)
    entry.metadata.type           !== undefined || (entry.metadata.type = '____')
    parent_entry.metadata.species !== undefined || (parent_entry.metadata.species = '____')

    if (entry.label !== null) {
        $('#ia-metadata-rowid').text('Part ID: ' + entry.label);
    } else {
        $('#ia-metadata-rowid').text('');
    }
    $('#ia-detection-part-viewpoint-1').val(entry.metadata.viewpoint1);
    $('#ia-detection-part-quality').val(entry.metadata.quality);

    // Update selected types
    $('#ia-detection-part-class').html('')
    $('#ia-detection-part-class')
        .append($("<option></option>")
            .attr("value", '____')
            .text('Unspecified'));

    // Collect primitives
    if ( ! parent_entry.metadata.species in species_part_dict) {
        parent_entry.metadata.species = '____'
    }
    if (species_part_dict[parent_entry.metadata.species].indexOf(entry.metadata.type) == -1) {
        species_part_dict[parent_entry.metadata.species].push(entry.metadata.type)
    }

    species = species_part_dict[parent_entry.metadata.species]
    for(var counter = 0; counter < species.length; counter++) {
        value = species[counter]
        console.log(value)
        if (value == '____') {
            continue
        }
        $('#ia-detection-part-class')
            .append($("<option></option>")
                .attr("value", value)
                .text(value));
    }

    $('#ia-detection-part-class option[value="' + entry.metadata.type + '"]').prop("selected", true);

    if(invalid) {
        $('.ia-detection-form-part-value').trigger('change');
    }

    update_label();
}

function fix_metadata_panels() {
    x = $('#ia-metadata-panel-annotation').height()
    y = $('#ia-metadata-panel-part').height()
    height = Math.max(x, y)
    if ( ! config_metadata){
      height = 0
    }
    $('#ia-metadata-panel-container').css({
        "height": height,
    })
    if (! config_parts) {
        $('.ia-detection-parts-toggle').css({
            'visibility': 'hidden',
        })
    }
}

function normalize_viewpoint(value1, value2, value3) {
    var viewpoint_strs = [];
    viewpoint_strs[-1] = null;
    viewpoint_strs[0] = 'Top';
    viewpoint_strs[1] = 'Bottom';
    viewpoint_strs[2] = 'Front';
    viewpoint_strs[3] = 'Back';
    viewpoint_strs[4] = 'Left';
    viewpoint_strs[5] = 'Right';

    if (value3 >= 0 && value1 == -1) {
        value3 = -1
        $("#ia-detection-" + state + "-viewpoint-3").val(value3)
    }
    if (value3 >= 0 && value2 == -1) {
        value3 = -1
        $("#ia-detection-" + state + "-viewpoint-3").val(value3)
    }
    if (value2 >= 0 && value1 == -1) {
        value2 = -1
        $("#ia-detection-" + state + "-viewpoint-2").val(value2)
    }

    if (value1 >= 0) {
        invalid1 = 2 * Math.floor(value1 / 2.0)
        invalid2 = invalid1 + 1
        $("#col-viewpoint-ticks-2-" + (invalid1 + 1)).css({
            opacity: 0.0,
        })
        $("#col-viewpoint-ticks-2-" + (invalid2 + 1)).css({
            opacity: 0.0,
        })
        $("#col-viewpoint-ticks-3-" + (invalid1 + 1)).css({
            opacity: 0.0,
        })
        $("#col-viewpoint-ticks-3-" + (invalid2 + 1)).css({
            opacity: 0.0,
        })

        if (value2 == invalid1 || value2 == invalid2) {
            value2 = -1
            value3 = -1
            $("#ia-detection-" + state + "-viewpoint-2").val(value2)
            $("#ia-detection-" + state + "-viewpoint-3").val(value2)
        }

        if (value2 >= 0) {
            invalid3 = 2 * Math.floor(value2 / 2.0)
            invalid4 = invalid3 + 1
            $("#col-viewpoint-ticks-3-" + (invalid3 + 1)).css({
                opacity: 0.0,
            })
            $("#col-viewpoint-ticks-3-" + (invalid4 + 1)).css({
                opacity: 0.0,
            })

            if (value3 == invalid1 || value3 == invalid2 || value3 == invalid3 || value3 == invalid4) {
                value3 = -1
                $("#ia-detection-" + state + "-viewpoint-3").val(value3)
            }
        }
    }

    tag1 = viewpoint_strs[value1]
    tag2 = viewpoint_strs[value2]
    tag3 = viewpoint_strs[value3]

    if (tag1 != null) {
        if (tag2 != null) {
            if (tag3 != null) {
                tag = tag1 + "-" + tag2 + "-" + tag3
            } else {
                tag = tag1 + "-" + tag2
            }
        } else {
            tag = tag1
        }
    } else {
        tag = "<i>Unspecified</i>"
    }

    return tag
}

function update_label() {
    state = states.active

    for (var index = 0; index <= 6; index++) {
        $("#col-viewpoint-ticks-2-" + index).css({
            opacity: 1.0,
        })
        $("#col-viewpoint-ticks-3-" + index).css({
            opacity: 1.0,
        })
    }

    value1 = parseFloat($("#ia-detection-" + state + "-viewpoint-1").val());
    value2 = parseFloat($("#ia-detection-" + state + "-viewpoint-2").val());
    value3 = parseFloat($("#ia-detection-" + state + "-viewpoint-3").val());

    tag = normalize_viewpoint(value1, value2, value3)
    $("#ia-detection-" + state + "-viewpoint-label").html(tag);

    var quality_strs = [];
    quality_strs[1] = 'Poor';
    quality_strs[2] = 'Good';

    value = parseFloat($("#ia-detection-" + state + "-quality").val());
    $("#ia-detection-" + state + "-quality-label").html(quality_strs[value]);
}

function add_species() {
    value = $('input[name="species-add"]').val()

    $('#ia-detection-annotation-class')
        .append($("<option></option>")
            .attr("value", value)
            .text(value));
    $('#ia-detection-annotation-class option[value="' + value + '"]').prop("selected", true);
    $('.ia-detection-form-annotation-value').trigger('change');
}

function add_part() {
    value = $('input[name="part-add"]').val()

    $('#ia-detection-part-class')
        .append($("<option></option>")
            .attr("value", value)
            .text(value));
    $('#ia-detection-part-class option[value="' + value + '"]').prop("selected", true);
    $('.ia-detection-form-part-value').trigger('change');
}

$(window).keydown(function(event) {
    key = event.which;

    if( ! hotkeys_global_disabled) {
        console.log(key)
        console.log(states.active)

        if (key == 17) {
            // Ctrl pressed
            $('.ia-detection-hotkey').show();
        } else if (key == 84 && ! single_mode) {
            // T pressed
            var element = $('#ia-detection-setting-toggle')
            element.prop('checked', !element.is(':checked')).trigger('change');
            // $('#ia-detection-setting-toggle').bootstrapToggle('toggle');
        } else if (key == 77) {
            // M pressed
            var element = $('#ia-detection-setting-orientation')
            element.prop('checked', !element.is(':checked')).trigger('change');
        } else if (key == 65) {
            // A pressed
            var element = $('#ia-detection-setting-parts-assignments')
            element.prop('checked', !element.is(':checked')).trigger('change');
        } else if (key == 83) {
            // S pressed
            var element = $('#ia-detection-setting-parts-show')
            element.prop('checked', !element.is(':checked')).trigger('change');
        } else if (key == 72 || key == 68) {
            // D pressed
            var element = $('#ia-detection-setting-parts-hide')
            element.prop('checked', !element.is(':checked')).trigger('change');
        } else if (key == 13) {
            // Enter key pressed
            $('input#ia-review-submit-accept').click();
        }
        else if (key == 32) {
            // // Space key pressed
            // $('input#ia-review-submit-clear').click();
            $('input#ia-review-submit-accept').click();
            event.preventDefault();
            return false;
        }
        else if (key == 80) {
            // P key pressed, follow previous link
            $('a#ia-review-previous')[0].click();
        }

        if ( ! hotkeys_disabled) {
            state = states.active
            console.log(state)
            if (state == null) {
                return
            }

            if (key == 81) {
                // Q key pressed, poor quality
                $("#ia-detection-" + state + "-quality").val(1).trigger('change');
            } else if (key == 87) {
                // W key pressed, good quality
                $("#ia-detection-" + state + "-quality").val(2).trigger('change');
            } else if (key == 69) {
                // E pressed
                $("#ia-detection-" + state + "-quality").val(0).trigger('change');
            } else if (key == 69) {
                // O pressed
                var element = $("#ia-detection-" + state + "-multiple")
                element.prop('checked', !element.is(':checked')).trigger('change');
            } else if (key == 79) {
                // E pressed
                if (state == "annotation") {
                    var element = $("#ia-detection-" + state + "-multiple")
                    element.prop('checked', !element.is(':checked')).trigger('change');
                } else {
                    $("#ia-detection-" + state + "-quality").val(0).trigger('change');
                }

            } else if (key == 73) {
                // I pressed
                var element = $("#ia-detection-" + state + "-interest")
                element.prop('checked', !element.is(':checked')).trigger('change');
            } else if (key == 67) {
                // C pressed
                var element = $("#ia-detection-" + state + "-class")
                var children = element.children('option')
                var length = children.length;
                var index = element.find("option:selected").index()
                // Increment
                if(event.shiftKey) {
                    index -= 1
                } else {
                    index += 1
                }
                index = index % length
                children.eq(index).prop('selected', true);
                $(".ia-detection-form-" + state + "-value").trigger('change');
            } else if ((48 <= key && key <= 55) || (96 <= key && key <= 103)) {
                // 48 == 46  == numeric key 0
                // 49 == 97  == numeric key 1
                // 50 == 98  == numeric key 2
                // 51 == 99  == numeric key 3
                // 52 == 100 == numeric key 4
                // 53 == 101 == numeric key 5
                // 54 == 102 == numeric key 6
                // 55 == 103 == numeric key 7
                // 56 == 104 == numeric key 8
                // 57 == 105 == numeric key 9

                value1 = parseFloat($("#ia-detection-" + state + "-viewpoint-1").val());
                value2 = parseFloat($("#ia-detection-" + state + "-viewpoint-2").val());

                if (value1 == -1 && event.altKey) {
                    return
                }

                if (!axis2 || value1 == -1 || event.shiftKey || state == "part") {
                    element = "#ia-detection-" + state + "-viewpoint-1"
                } else if (!axis3 || value2 == -1 || event.altKey) {
                    element = "#ia-detection-" + state + "-viewpoint-2"
                } else {
                    element = "#ia-detection-" + state + "-viewpoint-3"
                }

                if (key == 48 || key == 96) {
                    value = -1
                } else if (49 <= key && key <= 56) {
                    value = key - 49;
                } else {
                    value = key - 97;
                }

                $(element).val(value).trigger('change');
            }
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
