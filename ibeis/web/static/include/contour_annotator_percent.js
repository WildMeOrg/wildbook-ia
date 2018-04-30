(function() {
    var ContourSelector

    ContourSelector = (function() {
        function ContourSelector(frame, options) {
            var options

            options                     !== undefined || (options = {})
            options.prefix              !== undefined || (options.prefix = "")
            options.ids                 !== undefined || (options.ids = {})
            options.ids.canvas          !== undefined || (options.ids.canvas = options.prefix + "contour-selector-canvas")
            options.ids.begin           !== undefined || (options.ids.begin = options.prefix + "contour-selector-begin")
            options.ids.end             !== undefined || (options.ids.end = options.prefix + "contour-selector-end")
            options.classes             !== undefined || (options.classes = {})
            options.classes.radius      !== undefined || (options.classes.point = options.prefix + "contour-selector-segment-radius")
            options.classes.segment     !== undefined || (options.classes.point = options.prefix + "contour-selector-segment-point")
            options.colors              !== undefined || (options.colors = {})
            options.colors.begin        !== undefined || (options.colors.begin   = "#7FFF7F")
            options.colors.radius       !== undefined || (options.colors.radius  = "#333333")
            options.colors.segment      !== undefined || (options.colors.segment = "#2E63FF")
            options.colors.end          !== undefined || (options.colors.end     = "#EB2A18")
            options.size                !== undefined || (options.size = {})
            options.size.radius         !== undefined || (options.size.radius = 30)
            options.size.segment        !== undefined || (options.size.segment = 4)
            options.transparency        !== undefined || (options.transparency = {})
            options.transparency.radius !== undefined || (options.transparency.radius = 0.3)
            options.transparency.radius !== undefined || (options.transparency.segment = 1.0)
            options.limits.bounds       !== undefined || (options.limits.bounds = {})
            options.limits.bounds.x     !== undefined || (options.limits.bounds.x = {})
            options.limits.bounds.x.min !== undefined || (options.limits.bounds.x.min = 0)
            options.limits.bounds.x.max !== undefined || (options.limits.bounds.x.max = 0)
            options.limits.bounds.y     !== undefined || (options.limits.bounds.y = {})
            options.limits.bounds.y.min !== undefined || (options.limits.bounds.y.min = 0)
            options.limits.bounds.y.max !== undefined || (options.limits.bounds.y.max = 0)
            options.debug               !== undefined || (options.debug = false)

            // Global attributes
            this.options = options

            this.points = {
                segment: [],
            }
            this.elements = {
                segment: [],
            }

            this.ctx = null

            // Create objects
            this.create_selector_elements(frame)
        }

        ContourSelector.prototype.create_selector_elements = function(frame) {
            var width, height, endpoint_style

            // Define the image frame as an element
            this.elements.frame = frame

            width = this.elements.frame.width()
            height = this.elements.frame.height()

            canvas_style = {
                "opacity": this.options.transparency.radius,
                "width": width + "px",
                "height": height + "px",
            }

            // Create the draw canvas
            this.elements.canvas = $('<canvas width="' + width + '" height="' + height + '" id="' + this.options.ids.canvas + '"></canvas>')
            this.elements.canvas.css(canvas_style)
            this.elements.frame.append(this.elements.canvas)

            // Create the context
            this.ctx = this.elements.canvas[0].getContext('2d')

            // Begin point
            endpoint_style = {
                "top": "0.0px",
                "left": "0.0px",
                "width": "14px",
                "height": "14px",
                "position": "absolute",
                "border-radius": "14px",
                "border": "1px solid #fff",
                "margin-top": "-8px",
                "margin-left": "-8px",
                "z-index": "2",
            }

            // Begin endpoint
            this.elements.begin = $('<div id="' + this.options.ids.begin + '"></div>')
            this.elements.begin.css(endpoint_style)
            this.elements.begin.css({
                "background-color": this.options.colors.begin,
            })
            this.elements.frame.append(this.elements.begin)
            this.elements.begin.hide()

            // End endpoint
            this.elements.end = $('<div id="' + this.options.ids.begin + '"></div>')
            this.elements.end.css(endpoint_style)
            this.elements.end.css({
                "background-color": this.options.colors.end,
            })
            this.elements.frame.append(this.elements.end)
            this.elements.end.hide()
        }

        ContourSelector.prototype.check_boundaries = function(event) {
            var offset, point, bounds

            offset = this.elements.frame.offset()
            point = {
                x: event.pageX - offset.left,
                y: event.pageY - offset.top,
            }
            bounds = {
                x: {
                    min: 0,
                    max: 0,
                },
                y: {
                    min: 0,
                    max: 0,
                },
            }

            // Check lower boundaries
            point.x = Math.max(point.x, 0 - bounds.x.min)
            point.y = Math.max(point.y, 0 - bounds.y.min)
            // Check upper boundaries
            point.x = Math.min(point.x, this.elements.frame.width() - 1 + bounds.x.max)
            point.y = Math.min(point.y, this.elements.frame.height() - 1 + bounds.y.max)
            return point
        }

        ContourSelector.prototype.start = function(event) {
            this.points.begin = this.check_boundaries(event)
            this.points.cursor = this.points.begin

            this.ctx.clearRect(0, 0, this.elements.canvas.width(), this.elements.canvas.height());

            for(var index = 0; index < this.elements.segment.length; index++) {
                this.elements.segment[index].detach()
            }

            this.points.segment = []
            this.elements.segment = []

            console.log("STARTED")
            console.log(this.points.cursor)

            // Show selectors, as needed
            this.elements.end.hide()

            this.elements.begin.show()
            this.elements.begin.css({
                "top": this.points.begin.y + "px",
                "left": this.points.begin.x + "px",
            })

            // Set global cursor mode and prevent highlight selection
            $("body").css({
                "cursor": "crosshair"
            })

            document.onselectstart = function() {
                return false
            }

            // Don't refresh until we get new coordinates from the ContourAnnotator
            // this.refresh(event)
        }

        ContourSelector.prototype.update_cursor = function(event) {
            var size, size_half, radius, radius_style, segment, segment_style

            // Update the selector based on the current cursor location
            this.points.cursor = this.check_boundaries(event)
            this.points.segment.push(
                [this.points.cursor.x, this.points.cursor.y, this.options.size.radius]
            )

            console.log("UPDATE")

            size = this.options.size.radius
            this.ctx.beginPath()
            this.ctx.fillStyle = this.options.colors.radius
            this.ctx.ellipse(this.points.cursor.x, this.points.cursor.y, size, size, 0, 0, 2 * Math.PI)
            this.ctx.fill()

            size = this.options.size.segment
            size_half = Math.floor(size / 2.0)
            segment_style = {
                "top": this.points.cursor.y + "px",
                "left": this.points.cursor.x + "px",
                "width": size + "px",
                "height": size + "px",
                "position": "absolute",
                "border-radius": size + "px",
                "margin-top": "-" + size_half + "px",
                "margin-left": "-" + size_half + "px",
                "background-color": this.options.colors.segment,
                "opacity": this.options.transparency.segment,
                "z-index": "1",
            }

            // Begin segment center point
            segment = $('<div class="' + this.options.classes.segment + '"></div>')
            segment.css(segment_style)
            this.elements.frame.append(segment)
            this.elements.segment.push(segment)
        }

        ContourSelector.prototype.finish = function(event) {
            var data

            this.points.end = this.check_boundaries(event)
            this.points.cursor = this.points.end

            console.log("FINISHED")
            console.log(this.points.cursor)

            this.elements.end.show()
            this.elements.end.css({
                "top": this.points.end.y + "px",
                "left": this.points.end.x + "px",
            })

            // If there is a null event, don't do much
            if (event == null) {
                data = null
            } else {
                data = this.get_selector_data(event)
            }

            // Reset the global cursor mode and re-allow highlight selection
            $("body").css({
                "cursor": "default"
            })
            document.onselectstart = null

            return data
        }

        ContourSelector.prototype.get_selector_data = function(event) {
            var width, height, data, diff_x, diff_y

            // Create the selector data object
            data = {
                points: this.segment
            }

            // Get the current width and height of the image to calculate percentages
            width = this.elements.frame.width()
            height = this.elements.frame.height()

            // TODO: Overwrite absolute pixel values with percentage values
            return data
        }

        ContourSelector.prototype.refresh = function(event) {
            var data, css_theta, css_rotate

            data = this.get_selector_data(event)
        }

        return ContourSelector
    })()

    this.ContourAnnotator = (function() {
        function ContourAnnotator(url, options) {
            var options

            options                              !== undefined || (options = {})
            options.prefix                       !== undefined || (options.prefix = "")
            options.ids                          !== undefined || (options.ids = {})
            options.ids.container                !== undefined || (options.ids.container = options.prefix + "contour-annotator-container")
            options.props                        !== undefined || (options.props = {})
            options.classes                      !== undefined || (options.classes = {})
            options.classes.bbox                 !== undefined || (options.classes.bbox = options.prefix + "contour-annotator-bbox")
            options.colors                       !== undefined || (options.colors = {})
            options.colors.default               !== undefined || (options.colors.default = "#7FFF7F")
            options.actions                      !== undefined || (options.actions = {})
            options.hotkeys                      !== undefined || (options.hotkeys = {})
            options.hotkeys.enabled              !== undefined || (options.hotkeys.enabled = true)
            options.hotkeys.delete               !== undefined || (options.hotkeys.delete = [75, 8])
            options.hotkeys.zoom                 !== undefined || (options.hotkeys.zoom   = [90])
            options.limits                       !== undefined || (options.limits = {})
            options.limits.frame                 !== undefined || (options.limits.frame = {})
            options.limits.frame.width           !== undefined || (options.limits.frame.width =1000)
            options.limits.frame.height          !== undefined || (options.limits.frame.height = 700)
            options.limits.bounds                !== undefined || (options.limits.bounds = {})
            options.limits.bounds.x              !== undefined || (options.limits.bounds.x = {})
            options.limits.bounds.x.min          !== undefined || (options.limits.bounds.x.min = 50)
            options.limits.bounds.x.max          !== undefined || (options.limits.bounds.x.max = 50)
            options.limits.bounds.y              !== undefined || (options.limits.bounds.y = {})
            options.limits.bounds.y.min          !== undefined || (options.limits.bounds.y.min = 50)
            options.limits.bounds.y.max          !== undefined || (options.limits.bounds.y.max = 50)
            options.confirm                      !== undefined || (options.confirm = {})
            options.confirm.delete               !== undefined || (options.confirm.delete = true)
            options.callbacks                    !== undefined || (options.callbacks = {})
            options.callbacks.onload             !== undefined || (options.callbacks.onload = null)
            options.callbacks.onadd              !== undefined || (options.callbacks.onadd = null)
            options.callbacks.onselector         !== undefined || (options.callbacks.onselector = null)
            options.callbacks.onchange           !== undefined || (options.callbacks.onchange = null)
            options.callbacks.onhover            !== undefined || (options.callbacks.onhover = null)
            options.callbacks.onfocus            !== undefined || (options.callbacks.onfocus = null)
            options.callbacks.ondelete           !== undefined || (options.callbacks.ondelete = null)
            options.debug                        !== undefined || (options.debug = false)

            // Global attributes
            this.options = options

            this.entries = []
            this.elements = {
                entries: [],
            }

            // Keep track of the current state of the annotator
            this.state = {
                mode:     "free",
                inside:   false,
                which:    null,
                zoom:     false,
                anchors:  {},
            }

            // Create the annotator elements
            this.create_annotator_elements(url)
        }

        ContourAnnotator.prototype.create_annotator_elements = function(url) {
            var cta

            cta = this

            // Define the container
            this.elements.container = $("#" + this.options.ids.container)
            this.elements.container.css({
                "width": "100%",
                "margin-left": "auto",
                "margin-right": "auto",
            })

            // Create the image frame
            this.elements.frame = $('<div class="' + this.options.prefix + 'contour-annotator-frame"></div>')
            this.elements.frame.css({
                "position": "relative",
                "width": "100%",
                "height": "100%",
                "border": "#333333 solid 1px",
                "margin": "0px auto",
                "background-size": "100% auto",
                "background-repeat": "no-repeat",
                "overflow": "hidden",
                "cursor": "crosshair",
            })
            this.elements.container.append(this.elements.frame)

            ////////////////////////////////////////////////////////////////////////////////////

            // Create a new Javascript Image object and load the src provided by the user
            this.elements.image = new Image()
            this.elements.image.src = url

            // Create events to load the rest of the data, but only if the image loads
            this.elements.image.onload = function() {
                // Set the background image of the image frame to be the iamge src
                cta.elements.frame.css({
                    "background-image": "url(\"" + this.src + "\")",
                })

                // This is the last stage of the constructor's initialization
                cta.register_global_key_bindings()
                cta.register_global_event_bindings()

                // Resize the image to the window
                cta.resize()

                // We can now create the selector object after we have resized the view
                cta.cts = new ContourSelector(cta.elements.frame, cta.options)

                // Trigger onload
                if (cta.options.callbacks.onload != null) {
                    return cta.options.callbacks.onload()
                }
            }

            // If the image failed to load, add a message to the console.
            this.elements.image.onerror = function() {
                console.log.text("[ContourAnnotator] Invalid image URL provided: " + this.url)
            }
        }

        ContourAnnotator.prototype.delete_entry_interaction = function(index) {
            entry = cta.entries[index]

            if (this.options.confirm.delete && entry.parent == null && entry.label != null) {
                response = confirm("Are you sure you want to delete this box?")
                if ( ! response) {
                    return
                }
            }

            cta.delete_entry(index)
        }

        ContourAnnotator.prototype.register_global_key_bindings = function(selector, options) {
            var cta

            cta = this

            $(window).keydown(function(event) {
                var key, hotkeys_movement

                key = event.which

                // Shift key to zoom (un-zoom even if hotkeys are disabled)
                if (cta.options.hotkeys.zoom.indexOf(key) != -1) {
                    if (cta.state.zoom) {
                        cta.zoom_finish()
                    } else {
                        if (cta.options.hotkeys.enabled) {
                            cta.zoom_start()
                        }
                    }
                }

                if (cta.options.hotkeys.enabled) {

                    // Delete key pressed
                    if (cta.options.hotkeys.delete.indexOf(key) != -1) {
                        if (cta.state.mode == "selector") {
                            cta.selector_cancel(event)
                        } else {
                            cta.delete_entry_interaction(cta.state.hover)
                        }
                    }
                }
            })
        }

        ContourAnnotator.prototype.register_global_event_bindings = function(selector, options) {
            var cta

            cta = this

            // Resize the container dynamically, as needed, when the window resizes
            $(window).bind("resize", function() {
                cta.resize()
            })

            // Catch when a mouse down event happens inside (and only inside) the container
            this.elements.container.mousedown(function(event) {
                cta.state.inside = true
            })

            this.elements.container.contextmenu(function(event) {
                event.preventDefault()
            })

            // Catch the event when the mouse moves
            $(window).mousemove(function(event) {
                if (cta.state.mode == "selector") {
                    // Update the active ContourSelector location
                    cta.selector_update(event)
                }
            })

            // Catch the mouse up event (wherever it may appear on the screen)
            $(window).mouseup(function(event) {
                if (event.which == 1) {
                    if (cta.state.mode == "free" && cta.state.inside) {
                        cta.selector_start(event)
                    } else if (cta.state.mode == "selector") {
                        cta.selector_finish(event)
                    }
                }

                cta.state.inside = false
            })
        }

        ContourAnnotator.prototype.resize = function() {
            var w1, h1, w2, h2, offset, left, margin

            zoom = this.state.zoom
            margin = 5

            // Get the proportions of the image and
            w1 = this.elements.image.width
            h1 = this.elements.image.height

            if (zoom) {
                w2 = $(window).width() - 2 * margin
            } else {
                w2 = this.elements.container.width()
                limit1 = this.options.limits.frame.width
                limit2 = (this.options.limits.frame.height / h1) * w1
                w2 = Math.min(w2, this.options.limits.frame.width, limit2)
            }

            h2 = (w2 / w1) * h1

            if (zoom) {
                offset = this.elements.container.offset()
                left = -1.0 * offset.left
                left += margin
                left = left + "px"
            } else {
                left = ""
            }

            this.elements.frame.css({
                "width": w2 + "px",
                "height": h2 + "px",
                "left": left
            })
            this.elements.container.css({
                "height": h2 + "px",
            })

            // Update console
            this.refresh()
        }

        ContourAnnotator.prototype.zoom_start = function() {
            this.state.zoom = true
            this.resize()
        }

        ContourAnnotator.prototype.zoom_finish = function() {
            this.state.zoom = false
            this.resize()
        }

        ContourAnnotator.prototype.refresh = function() {
            if (this.options.callbacks.onchange != null) {
                return this.options.callbacks.onchange(this.entries)
            }
        }

        ContourAnnotator.prototype.add_entry = function(entry, notify) {
            var invalid, width, height, element, css_no_select, css_margin, css_handles, configurations, handle, index

            // Check the notify boolean variable
            notify !== undefined || (notify = false)

            // Check the required bounding box location dimensions are specified
            invalid = false
            entry.points !== undefined || (invalid = true)

            if ( ! invalid) {
                // Check structure of entry.points

                if ( ! invalid) {
                    width = this.elements.frame.width()
                    height = this.elements.frame.height()
                } else {
                    console.log("[ContourAnnotator] Invalid entry provided: points are incomplete")
                    console.log(entry)
                    return
                }
            } else {
                console.log("[ContourAnnotator] Invalid entry provided: no points")
                console.log(entry)
                return
            }

            // Ensure the entry has the required optional structures
            entry.label !== undefined || (entry.label = null)
            entry.metadata !== undefined || (entry.metadata = {})

            console.log('[ContourAnnotator] Adding entry for:')
            console.log(entry)

            // Create the holder element
            element = {}
            css_no_select = {
                "user-select": "none",
                "-o-user-select": "none",
                "-moz-user-select": "none",
                "-khtml-user-select": "none",
                "-webkit-user-select": "none",
            }

            // Create the bbox container
            element.bbox = $('<div class="' + this.options.classes.bbox + '"></div>')
            element.bbox.css({
                "position": "absolute",
                "top": entry.percent.top + "%",
                "left": entry.percent.left + "%",
                "width": entry.percent.width + "%",
                "height": entry.percent.height + "%",
                "border": this.options.border.width + "px solid " + this.options.colors.default,
                "border-top": (this.options.border.width * 1.5) + "px dotted " + this.options.colors.default,
                "color": "#FFFFFF",
                "font-family": "monospace",
                "font-size": "0px",
            })
            element.bbox.css(css_no_select)
            this.elements.frame.append(element.bbox)

            // Add the entry JSON and also the jQuery elements
            this.entries.push(entry)
            this.elements.entries.push(element)

            // Set the label for the entry
            index = this.entries.length - 1

            // notify on adding entry from selection
            if (notify && this.options.callbacks.onadd != null) {
                this.options.callbacks.onadd(entry)
            }

            // Refresh the view now that the new bbox has been added
            this.refresh()

            return index
        }

        ContourAnnotator.prototype.delete_entry = function(index) {
            var indices, entry, element

            if (index == null) {
                return
            }

            // Establish the original indices list
            indices = []
            for (var counter = 0; counter < this.entries.length; counter++) {
                indices.push(counter)
            }

            // Slice out the JSON entry and HTML elements from their respective list
            entry = this.entries.splice(index, 1)

            // Perform the exact same operation on the indices
            indices.splice(index, 1)

            // Detatch the bbox HTML elements at index
            element = this.elements.entries.splice(index, 1)
            element[0].bbox.detach()
            if (element[0].assignment != null) {
                element[0].assignment.detach()
            }

            // Refresh the display
            this.refresh()

            // Trigger ondelete
            if (cta.options.callbacks.ondelete != null) {
                return cta.options.callbacks.ondelete(entry[0])
            }
        }

        ContourAnnotator.prototype.selector_start = function(event) {
            // Start the ContourSelector
            this.cts.start(event)
            this.state.mode = "selector"

            // notify on adding entry from selection
            if (this.options.callbacks.onselector != null) {
                return this.options.callbacks.onselector()
            }
        }

        ContourAnnotator.prototype.selector_update = function(event) {
            // Update the ContourSelector
            this.cts.update_cursor(event)
        }

        ContourAnnotator.prototype.selector_cancel = function(event) {
            if (this.state.mode == "selector") {
                // Get the entry data by finishing the ContourSelector
                this.cts.finish(event)

                // Release the mode to "free"
                this.state.mode = "free"
            }
        }

        ContourAnnotator.prototype.selector_finish = function(event) {
            var data, invalid, index

            // If we are in the "selector" mode, finish the ContourSelector and add the entry
            if (this.state.mode == "selector") {
                // Get the entry data by finishing the ContourSelector
                data = this.cts.finish(event)

                invalid = true

                // Add the returned entry data reported by the ContourSelector as a new entry
                if( ! invalid) {
                    index = this.add_entry(data, true)
                }

                // Release the mode to "free"
                this.state.mode = "free"
            }
        }

        return ContourAnnotator

    })()

}).call(this)