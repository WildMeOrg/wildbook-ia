/*

TODO
- Hold Shift to resize bbox around center
- Hold Shift to move bbox while also moving any inside sub-entries
- On Selector finish/cancel, hover the correct box if actually hovering
- Selector assignment line for adding
- Delete entry if released entirely outside the bounds of the image
- Change the parent of a subentry
- Shift to background in reverse (foreground)
- Minimum size of the anchors with percentage based on box area
- Zooming for very small regions

*/

(function() {
    var BBoxSelector

    function calculate_angle(pointa, pointb) {
        var xa, ya, xb, yb, x1, x2, y1, y2, x, y, angle

        // Get the points and coordinates of extrema
        xa = pointa.x
        ya = pointa.y
        xb = pointb.x
        yb = pointb.y
        x1 = Math.min(xa, xb)
        x2 = Math.max(xa, xb)
        y1 = Math.min(ya, yb)
        y2 = Math.max(ya, yb)
        x = x2 - x1
        y = y2 - y1

        // calculate the angle, preserving sign after arctangent
        if (xa < xb && ya < yb) {
            angle = Math.atan2(y, x)
        } else if (xa < xb && ya >= yb) {
            angle = Math.atan2(y, x)
            angle *= -1.0
        } else if (xa >= xb && ya < yb) {
            angle = Math.atan2(x, y)
            angle += 0.5 * Math.PI
        } else if (xa >= xb && ya >= yb) {
            angle = Math.atan2(y, x)
            angle += Math.PI
        }

        // Normalize the angle to be between 0 and 2*PI
        while (angle < 0) {
            angle += 2.0 * Math.PI
        }
        while (angle > 2.0 * Math.PI) {
            angle -= 2.0 * Math.PI
        }
        return angle
    }

    $.fn.swapWith = function(that) {
        // http://stackoverflow.com/a/38515050/1413865

        var $this = this
        var $that = $(that)

        // create temporary placeholder
        var $temp = $("<div>")

        // 3-step swap
        $this.before($temp)
        $that.before($this)
        $temp.after($that).remove()

        return $this
    }

    BBoxSelector = (function() {
        function BBoxSelector(frame, options) {
            var options

            options                     !== undefined || (options = {})
            options.prefix              !== undefined || (options.prefix = "")
            options.ids                 !== undefined || (options.ids = {})
            options.ids.rectangle       !== undefined || (options.ids.rectangle = options.prefix + "bbox-selector-rectangle")
            options.ids.diagonal        !== undefined || (options.ids.diagonal  = options.prefix + "bbox-selector-diagonal")
            options.colors              !== undefined || (options.colors = {})
            options.colors.subentry     !== undefined || (options.colors.subentry = "#444444")
            options.border              !== undefined || (options.border = {})
            options.border.color        !== undefined || (options.border.color = "#7FFF7F")
            options.border.width        !== undefined || (options.border.width = 2)
            options.limits.bounds       !== undefined || (options.limits.bounds = {})
            options.limits.bounds.x     !== undefined || (options.limits.bounds.x = {})
            options.limits.bounds.x.min !== undefined || (options.limits.bounds.x.min = 0)
            options.limits.bounds.x.max !== undefined || (options.limits.bounds.x.max = 0)
            options.limits.bounds.y     !== undefined || (options.limits.bounds.y = {})
            options.limits.bounds.y.min !== undefined || (options.limits.bounds.y.min = 0)
            options.limits.bounds.y.max !== undefined || (options.limits.bounds.y.max = 0)
            options.modes               !== undefined || (options.modes = ["rectangle", "diagonal", "diagonal2"])
            options.mode                !== undefined || (options.mode = "rectangle")
            options.debug               !== undefined || (options.debug = false)

            // Global attributes
            this.options = options

            this.points = {}
            this.elements = {}
            this.current_stage = 0 // Only used when the diagonal selector is used
            this.debug = []

            // Create objects
            this.create_selector_elements(frame)

            // Set (update) selector mode
            this.update_mode(this.options.mode)
        }

        BBoxSelector.prototype.create_selector_elements = function(frame) {
            var selector_style, debug_style

            // Define the image frame as an element
            this.elements.frame = frame

            // Default style inherited by both selectors
            selector_style = {
                "border": this.options.border.width + "px dotted " + this.options.border.color,
                "position": "absolute",
                "transform-origin": "0% 0%",
            }

            // Create rectangle selector object
            this.elements.rectangle = $('<div id="' + this.options.ids.rectangle + '"></div>')
            this.elements.rectangle.css(selector_style)
            this.elements.frame.append(this.elements.rectangle)
            this.elements.rectangle.hide()

            // Create diagonal selector object
            this.elements.diagonal = $('<div id="' + this.options.ids.diagonal + '"></div>')
            this.elements.diagonal.css(selector_style)
            this.elements.frame.append(this.elements.diagonal)
            this.elements.diagonal.hide()

            // Create the debug objects for visual purposes
            if (this.options.debug) {
                debug_style = {
                    "top": "0.0",
                    "left": "0.0",
                    "width": "10px",
                    "height": "10px",
                    "position": "absolute",
                    "border-radius": "10px",
                    "border": "1px solid #fff",
                    "margin-top": "-6px",
                    "margin-left": "-6px",
                }

                colors = ["red", "green", "blue"]
                for (var index = 0; index < colors.length; index++) {
                    this.debug.push($("<div></div>"))
                    this.debug[index].css(debug_style)
                    this.debug[index].css({
                        "background-color": colors[index],
                    })
                    this.elements.frame.append(this.debug)
                }
            }
        }

        BBoxSelector.prototype.update_mode = function(proposed_mode) {
            // The only valid modes are "rectangle" and "diagonal"
            switch (proposed_mode) {
                case "rectangle":
                case "diagonal":
                case "diagonal2":
                    if (this.options.modes.indexOf(proposed_mode) == -1) {
                        console.log("[BBoxSelector] User disabled selection mode provided: " + proposed_mode)
                    } else {
                        this.options.mode = proposed_mode
                    }
                    break
                default:
                    console.log("[BBoxSelector] Invalid selection mode provided: " + proposed_mode)
            }

            // Return the active mode
            return this.options.mode
        }

        BBoxSelector.prototype.check_boundaries = function(event) {
            var offset, point, bounds

            offset = this.elements.frame.offset()
            point = {
                x: event.pageX - offset.left,
                y: event.pageY - offset.top,
            }
            if (this.options.mode == "rectangle") {
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
            } else {
                bounds = this.options.limits.bounds
            }

            // Check lower boundaries
            point.x = Math.max(point.x, 0 - bounds.x.min)
            point.y = Math.max(point.y, 0 - bounds.y.min)
            // Check upper boundaries
            point.x = Math.min(point.x, this.elements.frame.width() - 1 + bounds.x.max)
            point.y = Math.min(point.y, this.elements.frame.height() - 1 + bounds.y.max)
            return point
        }

        BBoxSelector.prototype.start = function(event, subentry) {
            var css_subentry

            subentry !== undefined || (subentry = false)

            this.points.origin = this.check_boundaries(event)
            this.points.middle = {
                x: NaN,
                y: NaN,
            }
            this.points.cursor = this.points.origin

            this.current_stage = 0

            // Show selectors, as needed
            if (this.options.mode == "rectangle") {
                this.elements.rectangle.show()
                this.elements.diagonal.hide()
                this.elements.rectangle.css({
                    "opacity": "0.0",
                })
            } else {
                this.elements.rectangle.hide()
                this.elements.diagonal.show()
                this.elements.diagonal.css({
                    "opacity": "0.0",
                    "width": "0px",
                })
            }

            if(subentry) {
                css_subentry = {
                    "outline": this.options.border.width + "px solid " + this.options.colors.subentry,
                }
            } else {
                css_subentry = {
                    "outline": "none",
                }
            }
            this.elements.rectangle.css(css_subentry)
            this.elements.diagonal.css(css_subentry)

            // Set global cursor mode and prevent highlight selection
            $("body").css({
                "cursor": "crosshair"
            })
            document.onselectstart = function() {
                return false
            }

            // Don't refresh until we get new coordinates from the BBoxAnnotator
            // this.refresh(event)
        }

        BBoxSelector.prototype.stage = function(event) {
            var finish

            finish = false

            // Used only for a multi-staging selection (i.e. diagonal)
            if (this.options.mode == "rectangle") {
                finish = true
            } else {
                if (this.current_stage == 0) {
                    this.current_stage = 1 // We only want to stage once
                    this.elements.rectangle.show()
                } else {
                    finish = true
                }
            }
            this.refresh(event)

            return finish
        }


        BBoxSelector.prototype.finish = function(event) {
            var data

            // If there is a null event, don't do much
            if (event == null) {
                data = null
            } else {
                data = this.get_selector_data(event)
            }

            // Hide selectors
            this.elements.rectangle.hide()
            this.elements.diagonal.hide()

            // Reset the global cursor mode and re-allow highlight selection
            $("body").css({
                "cursor": "default"
            })
            document.onselectstart = null

            return data
        }

        BBoxSelector.prototype.update_cursor = function(event) {
            // Update the selector based on the current cursor location
            this.points.cursor = this.check_boundaries(event)
            if (this.current_stage != 1) {
                this.points.middle = this.points.cursor
            }

            this.refresh(event)
        }

        BBoxSelector.prototype.get_selector_data = function(event) {
            var width, height, data, diff_x, diff_y

            // Create the selector data object
            data = {
                pixels: {
                    origin: this.points.origin,
                    middle: this.points.middle,
                    cursor: this.points.cursor,
                },
                percent: {},
                angles: {},
                metadata: {},
            }

            // Calculate the raw pixel locations and angles
            data.pixels.xtl = Math.min(data.pixels.origin.x, data.pixels.middle.x)
            data.pixels.ytl = Math.min(data.pixels.origin.y, data.pixels.middle.y)
            data.pixels.xbr = Math.max(data.pixels.origin.x, data.pixels.middle.x)
            data.pixels.ybr = Math.max(data.pixels.origin.y, data.pixels.middle.y)
            data.pixels.width = data.pixels.xbr - data.pixels.xtl
            data.pixels.height = data.pixels.ybr - data.pixels.ytl

            data.pixels.hypo = Math.sqrt(Math.pow(data.pixels.width, 2) + Math.pow(data.pixels.height, 2))
            diff_x = data.pixels.middle.x - data.pixels.cursor.x
            diff_y = data.pixels.middle.y - data.pixels.cursor.y
            data.pixels.right = Math.sqrt(Math.pow(diff_x, 2) + Math.pow(diff_y, 2))

            data.angles.origin = calculate_angle(data.pixels.origin, data.pixels.middle)
            data.angles.middle = calculate_angle(data.pixels.middle, data.pixels.cursor)
            data.angles.right = data.angles.origin - Math.PI * 0.5
            data.angles.theta = data.angles.origin

            // Calculate the left, top, width, and height in pixel space
            if (this.options.mode == "rectangle") {
                data.pixels.left = data.pixels.xtl
                data.pixels.top = data.pixels.ytl
                data.angles.theta = 0.0
            } else {
                var diff_angle, diff_len, position, center

                if (this.current_stage > 0) {
                    // Calculate the perpendicular angle and offset from the first segment
                    diff_angle = data.angles.middle - data.angles.origin
                    diff_len = Math.sin(diff_angle) * data.pixels.right
                    data.pixels.adjacent = Math.abs(diff_len)
                    data.pixels.correction = {
                        x: Math.cos(data.angles.right) * data.pixels.adjacent,
                        y: Math.sin(data.angles.right) * data.pixels.adjacent,
                    }

                    data.angles.fixed = diff_len > 0
                    if (data.angles.fixed) {

                        data.angles.theta += Math.PI
                    }
                }

                // Calculate the center of the first segment
                centers = {}
                centers.hypo = {
                    x: data.pixels.xtl + data.pixels.width * 0.5,
                    y: data.pixels.ytl + data.pixels.height * 0.5,
                }
                centers.rect = {
                    x: data.pixels.xtl + this.elements.rectangle.width() * 0.5,
                    y: data.pixels.ytl + this.elements.rectangle.height() * 0.5,
                }

                // Rectify the two centers as the rotation is happening around the center
                centers.offset = {
                    x: centers.hypo.x - centers.rect.x,
                    y: centers.hypo.y - centers.rect.y,
                }

                if (this.debug.length > 0) {
                    this.debug[0].css({
                        "top": data.pixels.ytl + "px",
                        "left": data.pixels.xtl + "px",
                    })
                    this.debug[1].css({
                        "top": centers.hypo.y + "px",
                        "left": centers.hypo.x + "px",
                    })
                    this.debug[2].css({
                        "top": centers.rect.y + "px",
                        "left": centers.rect.x + "px",
                    })
                }

                // Normalize the pixel locations based on center rotation
                data.pixels.left = data.pixels.xtl + centers.offset.x
                data.pixels.top = data.pixels.ytl + centers.offset.y
                data.pixels.width = this.elements.rectangle.width()
                data.pixels.height = this.elements.rectangle.height()
            }

            // Get the current width and height of the image to calculate percentages
            width = this.elements.frame.width()
            height = this.elements.frame.height()

            // Calculate the final values in percentage space
            data.percent.left = 100.0 * data.pixels.left / width
            data.percent.top = 100.0 * data.pixels.top / height
            data.percent.width = 100.0 * data.pixels.width / width
            data.percent.height = 100.0 * data.pixels.height / height

            // Normalize the theta angle range to be between 0 and 2*PI
            while (data.angles.theta < 0) {
                data.angles.theta += 2.0 * Math.PI
            }
            while (data.angles.theta > 2.0 * Math.PI) {
                data.angles.theta -= 2.0 * Math.PI
            }

            return data
        }

        BBoxSelector.prototype.refresh = function(event) {
            var data, css_theta, css_rotate

            data = this.get_selector_data(event)

            if (this.options.mode == "rectangle") {
                css_theta = data.angles.theta

                // Update the appearance of the rectangle selection using percentages
                this.elements.rectangle.css({
                    "top": data.percent.top + "%",
                    "left": data.percent.left + "%",
                    "width": data.percent.width + "%",
                    "height": data.percent.height + "%",
                    "opacity": "1.0",
                })
            } else {
                css_theta = data.angles.origin

                if (this.current_stage == 0) {
                    // First stage of the diagonal selection: drawing the hypotenuse line
                    this.elements.diagonal.css({
                        "top": data.pixels.origin.y + "px",
                        "left": data.pixels.origin.x + "px",
                        "width": data.pixels.hypo + "px",
                        "opacity": "1.0",
                    })
                    this.elements.rectangle.css({
                        "top": data.pixels.origin.y + "px",
                        "left": data.pixels.origin.x + "px",
                        "width": data.pixels.hypo + "px",
                        "height": 0 + "px",
                    })
                } else {
                    // Second stage of the diagonal selection: expanding the rectangle
                    this.elements.diagonal.css({
                        "opacity": "0.5",
                    })
                    this.elements.rectangle.css({
                        "top": (data.pixels.origin.y + data.pixels.correction.y) + "px",
                        "left": (data.pixels.origin.x + data.pixels.correction.x) + "px",
                        "width": data.pixels.hypo + "px",
                        "height": 2.0 * data.pixels.adjacent + "px",
                    })
                }
            }

            // Rotate the boxes based on the appropriate values
            css_rotate = {
                "transform": "rotate(" + css_theta + "rad)",
                "msTransform": "rotate(" + css_theta + "rad)",
                "-o-transform": "rotate(" + css_theta + "rad)",
                "-ms-transform": "rotate(" + css_theta + "rad)",
                "-moz-transform": "rotate(" + css_theta + "rad)",
                "-sand-transform": "rotate(" + css_theta + "rad)",
                "-webkit-transform": "rotate(" + css_theta + "rad)",
            }

            this.elements.rectangle.css(css_rotate)
            this.elements.diagonal.css(css_rotate)
        }

        return BBoxSelector
    })()

    this.BBoxAnnotator = (function() {
        function BBoxAnnotator(url, options) {
            var options

            options                              !== undefined || (options = {})
            options.prefix                       !== undefined || (options.prefix = "")
            options.ids                          !== undefined || (options.ids = {})
            options.ids.container                !== undefined || (options.ids.container      = options.prefix + "bbox-annotator-container")
            options.ids.resize                   !== undefined || (options.ids.resize         = options.prefix + "bbox-annotator-bbox-resize")
            options.props                        !== undefined || (options.props = {})
            options.props.rotate                 !== undefined || (options.props.rotate       = options.prefix + "bbox-annotator-bbox-rotate-handle")
            options.props.resize                 !== undefined || (options.props.resize       = options.prefix + "bbox-annotator-bbox-resize-handle")
            options.classes                      !== undefined || (options.classes = {})
            options.classes.bbox                 !== undefined || (options.classes.bbox       = options.prefix + "bbox-annotator-bbox")
            options.classes.assignment           !== undefined || (options.classes.assignment = options.prefix + "bbox-annotator-bbox-assignment")
            options.classes.label                !== undefined || (options.classes.label      = options.prefix + "bbox-annotator-bbox-label")
            options.classes.close                !== undefined || (options.classes.close      = options.prefix + "bbox-annotator-bbox-close")
            options.classes.rotate               !== undefined || (options.classes.rotate     = options.prefix + "bbox-annotator-bbox-rotate")
            options.classes.resize               !== undefined || (options.classes.resize     = options.prefix + "bbox-annotator-bbox-resize")
            options.colors                       !== undefined || (options.colors = {})
            options.colors.default               !== undefined || (options.colors.default     = "#7FFF7F")
            options.colors.hover                 !== undefined || (options.colors.hover       = "#F0AD4E")
            options.colors.focus                 !== undefined || (options.colors.focus       = "#EB2A18")
            options.colors.anchor                !== undefined || (options.colors.anchor      = "#EB2A18")
            options.colors.subentry              !== undefined || (options.colors.subentry    = "#444444")
            options.colors.highlighted           !== undefined || (options.colors.highlighted = "#2E63FF")
            options.colors.transparent           !== undefined || (options.colors.transparent = 0.25)
            options.actions                      !== undefined || (options.actions = {})
            options.actions.entry                !== undefined || (options.actions.entry = {})
            options.actions.entry.addition       !== undefined || (options.actions.entry.addition = true)
            options.actions.entry.parts          !== undefined || (options.actions.entry.parts = true)
            options.actions.entry.translation    !== undefined || (options.actions.entry.translation = true)
            options.actions.entry.scaling        !== undefined || (options.actions.entry.scaling = true)
            options.actions.entry.rotation       !== undefined || (options.actions.entry.rotation = true)
            options.actions.entry.deletion       !== undefined || (options.actions.entry.deletion = true)
            options.actions.entry.defocus        !== undefined || (options.actions.entry.defocus = true)
            options.actions.subentry             !== undefined || (options.actions.subentry = {})
            options.actions.subentry.addition    !== undefined || (options.actions.subentry.addition = true)
            options.actions.subentry.parts       !== undefined || (options.actions.subentry.parts = true)
            options.actions.subentry.translation !== undefined || (options.actions.subentry.translation = true)
            options.actions.subentry.scaling     !== undefined || (options.actions.subentry.scaling = true)
            options.actions.subentry.rotation    !== undefined || (options.actions.subentry.rotation = true)
            options.actions.subentry.deletion    !== undefined || (options.actions.subentry.deletion = true)
            options.hotkeys                      !== undefined || (options.hotkeys = {})
            options.hotkeys.enabled              !== undefined || (options.hotkeys.enabled = true)
            options.hotkeys.delete               !== undefined || (options.hotkeys.delete           = [8, 46, 75]) // 46 for Firefox
            options.hotkeys.exit                 !== undefined || (options.hotkeys.exit             = [27])
            options.hotkeys.zoom                 !== undefined || (options.hotkeys.zoom             = [90])
            options.hotkeys.background           !== undefined || (options.hotkeys.background       = [66])
            options.hotkeys.focus                !== undefined || (options.hotkeys.focus            = [70])
            options.hotkeys.counterclockwise     !== undefined || (options.hotkeys.counterclockwise = [76])
            options.hotkeys.clockwise            !== undefined || (options.hotkeys.clockwise        = [82])
            options.hotkeys.left                 !== undefined || (options.hotkeys.left             = [37])
            options.hotkeys.up                   !== undefined || (options.hotkeys.up               = [38])
            options.hotkeys.right                !== undefined || (options.hotkeys.right            = [39])
            options.hotkeys.down                 !== undefined || (options.hotkeys.down             = [40])
            options.handles                      !== undefined || (options.handles = {})
            options.handles.rotate               !== undefined || (options.handles.rotate = {})
            options.handles.rotate.enabled       !== undefined || (options.handles.rotate.enabled = true)
            options.handles.rotate.diameter      !== undefined || (options.handles.rotate.diameter = 10)
            options.handles.resize               !== undefined || (options.handles.resize = {})
            options.handles.resize.enabled       !== undefined || (options.handles.resize.enabled = true)
            options.handles.resize.diameter      !== undefined || (options.handles.resize.diameter = 8)
            options.handles.close                !== undefined || (options.handles.close = {})
            options.handles.close.enabled        !== undefined || (options.handles.close.enabled = false)
            options.border                       !== undefined || (options.border = {})
            options.border.color                 !== undefined || (options.border.color = options.colors.default)
            options.border.width                 !== undefined || (options.border.width = 2)
            options.steps                        !== undefined || (options.steps = {})
            options.steps.move                   !== undefined || (options.steps.move = 10.0)
            options.steps.rotate                 !== undefined || (options.steps.rotate = Math.PI / 4.0)
            options.subentries                   !== undefined || (options.subentries = {})
            options.subentries.visible           !== undefined || (options.subentries.visible = null)
            options.subentries.assignments       !== undefined || (options.subentries.assignments = false)
            options.subentries.label             !== undefined || (options.subentries.label = true)
            options.limits                       !== undefined || (options.limits = {})
            options.limits.frame                 !== undefined || (options.limits.frame = {})
            options.limits.frame.width           !== undefined || (options.limits.frame.width =1000)
            options.limits.frame.height          !== undefined || (options.limits.frame.height = 700)
            options.limits.entry                 !== undefined || (options.limits.entry = {})
            options.limits.entry.area            !== undefined || (options.limits.entry.area = 100)
            options.limits.entry.width           !== undefined || (options.limits.entry.width = 10)
            options.limits.entry.height          !== undefined || (options.limits.entry.height = 10)
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
            options.modes                        !== undefined || (options.modes = ["rectangle", "diagonal", "diagonal2"])
            options.mode                         !== undefined || (options.mode = "rectangle")
            options.debug                        !== undefined || (options.debug = false)

            // Global attributes
            this.options = options

            this.entries = []
            this.elements = {
                entries: [],
            }
            this.debug = []

            // Keep track of the current state of the annotator
            this.state = {
                mode:     "free",
                hover:    null,
                hover2:   null,
                subentry: null,
                focus:    null,
                focus2:   null,
                drag:     false,
                which:    null,
                inside:   false,
                zoom:     false,
                anchors:  {},
            }

            // Create the annotator elements
            this.create_annotator_elements(url)
        }

        BBoxAnnotator.prototype.create_annotator_elements = function(url) {
            var bba, debug_style

            bba = this

            // Define the container
            this.elements.container = $("#" + this.options.ids.container)
            this.elements.container.css({
                "width": "100%",
                "margin-left": "auto",
                "margin-right": "auto",
            })

            // Create the image frame
            this.elements.frame = $('<div class="ia-bbox-annotator-frame"></div>')
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

            // Create the debug objects for visual purposes
            if (this.options.debug) {
                debug_style = {
                    "left": "0.0",
                    "top": "0.0",
                    "width": "10px",
                    "height": "10px",
                    "position": "absolute",
                    "border-radius": "10px",
                    "border": "1px solid #fff",
                    "margin-top": "-6px",
                    "margin-left": "-6px",
                }

                colors = ["red", "green", "blue", "purple"]
                for (var index = 0; index < colors.length; index++) {
                    this.debug.push($("<div></div>"))
                    this.debug[index].css(debug_style)
                    this.debug[index].css({
                        "background-color": colors[index],
                    })
                    this.elements.frame.append(this.debug)
                }
            }

            ////////////////////////////////////////////////////////////////////////////////////

            // Create a new Javascript Image object and load the src provided by the user
            this.elements.image = new Image()
            this.elements.image.src = url

            // Create events to load the rest of the data, but only if the image loads
            this.elements.image.onload = function() {
                // Set the background image of the image frame to be the iamge src
                bba.elements.frame.css({
                    "background-image": "url(\"" + this.src + "\")",
                })

                // We can now create the selector object
                bba.bbs = new BBoxSelector(bba.elements.frame, bba.options)

                // This is the last stage of the constructor's initialization
                bba.register_global_key_bindings()
                bba.register_global_event_bindings()

                // Resize the image to the window
                bba.resize()

                // Trigger onload
                if (bba.options.callbacks.onload != null) {
                    return bba.options.callbacks.onload()
                }
            }

            // If the image failed to load, add a message to the console.
            this.elements.image.onerror = function() {
                console.log.text("[BBoxAnnotator] Invalid image URL provided: " + this.url)
            }
        }

        BBoxAnnotator.prototype.delete_entry_interaction = function(event, index) {
            entry = bba.entries[index]

            if(event.shiftKey && entry.parent == null) {
                if (this.options.actions.subentry.deletion == false) {
                    return
                }

                if (this.options.confirm.delete) {
                    response = confirm("Are you sure you want to delete all sub-entries?")
                     if ( ! response) {
                        return
                     }
                }

                // Recursively delete any sub-entries that belong to the parent index
                for (var index_entry = 0; index_entry < bba.entries.length; index_entry++) {
                    var parent

                    parent = bba.entries[index_entry].parent
                    if (parent != null && parent == index) {
                        index = bba.delete_entry(index_entry, index)
                        index_entry = 0
                    }
                }
            } else {
                if (entry.parent == null) {
                    if (this.options.actions.entry.deletion == false) {
                        return
                    }
                } else {
                    if (this.options.actions.subentry.deletion == false) {
                        return
                    }
                }

                if (this.options.confirm.delete && entry.parent == null && entry.label != null) {
                    response = confirm("Are you sure you want to delete this box?")
                    if ( ! response) {
                        return
                    }
                }

                // Recursively delete any sub-entries that belong to the parent index
                for (var index_entry = 0; index_entry < bba.entries.length; index_entry++) {
                    var parent

                    parent = bba.entries[index_entry].parent
                    if (parent != null && parent == index) {
                        index = bba.delete_entry(index_entry, index)
                        index_entry = 0
                    }
                }

                bba.delete_entry(index)
            }
        }

        BBoxAnnotator.prototype.register_global_key_bindings = function(selector, options) {
            var bba

            bba = this

            $(window).keydown(function(event) {
                var key, hotkeys_movement

                key = event.which

                // Shift key to zoom (un-zoom even if hotkeys are disabled)
                if (bba.options.hotkeys.zoom.indexOf(key) != -1) {
                    if (bba.state.zoom) {
                        bba.zoom_finish()
                    } else {
                        if (bba.options.hotkeys.enabled) {
                            bba.zoom_start()
                        }
                    }
                }

                if (bba.options.hotkeys.enabled) {

                    // Delete key pressed
                    if (bba.options.hotkeys.delete.indexOf(key) != -1) {
                        event.preventDefault() // For Firefox

                        if (bba.state.mode == "selector") {
                            bba.selector_cancel(event)
                        } else if (bba.state.mode == "drag") {
                            return
                        } else if (bba.state.mode == "rotate") {
                            return
                        } else if (bba.state.mode == "resize") {
                            return
                        } else if (bba.state.focus2 != null) {
                            bba.focus_entry(bba.state.focus2)
                            if(bba.state.hover == null) {
                                bba.subentries_style_visible(null)
                            }
                        } else if (bba.state.focus != null) {
                            if(bba.state.hover == null || bba.state.hover == bba.state.focus) {
                                bba.focus_entry(bba.state.focus)
                                if(bba.state.hover == null) {
                                    bba.subentries_style_visible(null)
                                }
                            } else if (bba.state.hover != null) {
                                bba.delete_entry_interaction(event, bba.state.hover)
                            }
                        } else {
                            bba.delete_entry_interaction(event, bba.state.hover)
                        }
                    }

                    // Exit key pressed
                    if (bba.options.hotkeys.exit.indexOf(key) != -1) {
                        if (bba.state.mode == "selector") {
                            bba.selector_cancel(event)
                        } else if (bba.state.mode == "drag") {
                            return
                        } else if (bba.state.mode == "rotate") {
                            return
                        } else if (bba.state.mode == "resize") {
                            return
                        } else if (bba.state.focus != null) {
                            bba.focus_entry(bba.state.focus)
                            if(bba.state.hover == null) {
                                bba.subentries_style_visible(null)
                            }
                        }
                    }

                    // F key to focus
                    if (bba.options.hotkeys.focus.indexOf(key) != -1) {
                        if (bba.state.hover != null) {
                            bba.focus_entry(bba.state.hover)
                        } else if (bba.state.focus != null) {
                            bba.focus_entry(bba.state.focus)
                            // If no longer hovering, hide entries
                            if(bba.state.hover == null) {
                                bba.subentries_style_visible(null)
                            }
                        }
                    }

                    // Modify bbox with keys (but only if nor focusing)
                    if (bba.state.focus == null || bba.state.focus != bba.state.hover) {
                        // Movement arrow keys for hovered bbox movement
                        hotkeys_movement = []
                        hotkeys_movement = hotkeys_movement.concat(bba.options.hotkeys.up)
                        hotkeys_movement = hotkeys_movement.concat(bba.options.hotkeys.down)
                        hotkeys_movement = hotkeys_movement.concat(bba.options.hotkeys.left)
                        hotkeys_movement = hotkeys_movement.concat(bba.options.hotkeys.right)
                        if (hotkeys_movement.indexOf(key) != -1) {
                            correction = {
                                x: 0.0,
                                y: 0.0,
                            }

                            if (bba.options.hotkeys.left.indexOf(key) != -1) {
                                // Left arrow
                                correction.x = -1.0
                            }
                            if (bba.options.hotkeys.up.indexOf(key) != -1) {
                                // Up arrow
                                correction.y = -1.0
                            }
                            if (bba.options.hotkeys.right.indexOf(key) != -1) {
                                // Right arrow
                                correction.x = 1.0
                            }
                            if (bba.options.hotkeys.down.indexOf(key) != -1) {
                                // Down arrow
                                correction.y = 1.0
                            }

                            // If the shift key is pressed, move by 10 pixels instead of 1
                            if (event.shiftKey) {
                                correction.x *= bba.options.steps.move
                                correction.y *= bba.options.steps.move
                            }

                            bba.move_entry_correction(event, correction)
                        }

                        // B key to send the bbox element to the back
                        if (bba.options.hotkeys.background.indexOf(key) != -1) {
                            bba.background_entry(event, bba.state.hover)
                        }

                        // L key to rotate (left by 90 degrees) the box
                        if (bba.options.hotkeys.counterclockwise.indexOf(key) != -1) {
                            bba.orient_entry(event, "left")
                        }

                        // R key to rotate (right by 90 degrees) the box
                        if (bba.options.hotkeys.clockwise.indexOf(key) != -1) {
                            bba.orient_entry(event, "right")
                        }
                    }
                }
            })
        }

        BBoxAnnotator.prototype.register_global_event_bindings = function(selector, options) {
            var bba

            bba = this

            // Resize the container dynamically, as needed, when the window resizes
            $(window).bind("resize", function() {
                bba.resize()
            })

            // Catch when a mouse down event happens inside (and only inside) the container
            this.elements.container.mousedown(function(event) {
                bba.state.inside = true

                if (bba.state.mode != "rotate" && bba.state.mode != "resize") {
                    bba.drag_start(event)
                }
            })

            this.elements.container.contextmenu(function(event) {
                event.preventDefault()
            })

            // Catch the event when the mouse moves
            $(window).mousemove(function(event) {
                if (bba.state.mode == "selector") {
                    // Update the active BBoxSelector location
                    bba.selector_update(event)
                } else if (bba.state.drag && bba.state.mode != "drag" && bba.state.mode != "magnet") {
                    // We are dragging the cursor, move the entry if hovered
                    if (bba.state.which == 3) {
                        bba.state.mode = "magnet"
                        element = bba.elements.entries[bba.state.hover]
                        entry = bba.entries[bba.state.hover]
                        // VERSION 1
                        resize_handle = element.resize[entry.closest]
                        bba.resize_start(resize_handle, event)
                        // VERSION 2
                        // bba.anchor_update(event, entry.closest)
                    } else if (bba.state.hover != null) {
                        bba.state.mode = "drag"

                        // Hide the close and rotate buttons
                        element = bba.elements.entries[bba.state.hover]

                        if(bba.state.focus == null || bba.state.focus != bba.state.hover) {
                            // Set the other entries to be transparent
                            bba.entries_style_transparent(bba.options.colors.transparent, bba.state.hover)

                            // If we have a parent, do not hide
                            entry = bba.entries[bba.state.hover]
                            if (entry.parent != null) {
                                element2 = bba.elements.entries[entry.parent]
                                element2.bbox.css({
                                    'opacity': 1.0,
                                })
                            }

                            element.label.hide()
                        }

                        if (!bba.options.debug) {
                            element.close.hide()
                            element.rotate.hide()
                            for (var key in element.resize) {
                                element.resize[key].hide()
                            }
                        }

                        // Update the anchor point
                        bba.anchor_update(event, null)
                    } else {
                        // This happens when we mousedown over nothing then drag over a bbox
                        bba.state.mode = "junk"
                        bba.drag_finish()
                    }
                }

                // If we are simply hovering, notify that the mouse has moved and highlight a correct anchor
                if (bba.state.mode == "free" && bba.state.hover != null) {
                    bba.anchor_entry(event)
                }

                // If we are modifying a bounding box via a drag, notify that the mouse has moved
                if (bba.state.mode == "drag") {
                    bba.move_entry(event)
                }

                // If we are rotating a bounding box via a drag, notify the bbox that the mouse has moved
                if (bba.state.mode == "rotate") {
                    bba.rotate_entry(event)
                }

                // If we are resizing a bounding box via a drag, notify the bbox that the mouse has moved
                if (bba.state.mode == "resize") {
                    bba.resize_entry(event)
                }

                // If we are resizing a bounding box via a magnet drag, notify the bbox that the mouse has moved
                if (bba.state.mode == "magnet") {
                    bba.resize_entry(event)
                }
            })

            // Catch the mouse up event (wherever it may appear on the screen)
            $(window).mouseup(function(event) {
                if (event.which == 1) {
                    if (bba.state.mode == "free" && bba.state.inside) {
                        if(bba.state.focus2 == null) {
                            bba.selector_start(event)
                        }
                    } else if (bba.state.mode == "selector") {
                        bba.selector_stage(event)
                    } else if (bba.state.mode == "drag") {

                        if (bba.state.hover != null) {
                            entry = bba.entries[bba.state.hover]
                            element = bba.elements.entries[bba.state.hover]

                            if (entry !== undefined && element !== undefined) {
                                bba.label_entry(bba.state.hover, entry.label)

                                if(bba.state.focus == null || bba.state.focus != bba.state.hover) {
                                    if(bba.state.focus2 == null) {
                                        // Set the other entries to be transparent
                                        bba.entries_style_transparent(1.0)

                                        if (bba.options.handles.close.enabled) {
                                            element.close.show()
                                        }
                                        if (bba.options.handles.rotate.enabled) {
                                            element.rotate.show()
                                        }
                                        for (var key in element.resize) {
                                            if (bba.options.handles.resize.enabled) {
                                                element.resize[key].show()
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        bba.state.mode = "free"
                    } else if (bba.state.mode == "rotate") {
                        var element

                        // Set the other entries to be transparent
                        bba.entries_style_transparent(1.0)

                        if (bba.state.hover != null) {
                            entry = bba.entries[bba.state.hover]
                            element = bba.elements.entries[bba.state.hover]

                            if (entry !== undefined && element !== undefined) {

                                bba.label_entry(bba.state.hover, entry.label)
                                if (bba.options.handles.close.enabled) {
                                    element.close.show()
                                }
                                for (var key in element.resize) {
                                    if (bba.options.handles.resize.enabled) {
                                        element.resize[key].show()
                                    }
                                }
                            }
                        }

                        document.onselectstart = null
                        bba.state.mode = "free"
                    } else if (bba.state.mode == "resize") {
                        bba.resize_finish(event)
                    } else if (bba.state.mode == "close") {
                        bba.state.mode = "free"
                    } else if (bba.state.mode == "junk") {
                        bba.state.mode = "free"
                    }
                } else if (event.which == 3) {
                    if (bba.state.mode == "magnet") {
                        bba.resize_finish(event)
                    } else {
                        if (bba.state.hover != null) {
                            bba.focus_entry(bba.state.hover)
                        } else if (bba.state.focus != null) {
                            bba.focus_entry(bba.state.focus)
                        }
                    }
                }

                bba.drag_finish()
                bba.state.inside = false
            })
        }

        BBoxAnnotator.prototype.resize = function() {
            var w1, h1, w2, h2, offset, left, margin

            zoom = this.state.zoom
            margin = 5

            // Get the proportions of the image and
            width = $(window).width()
            height = $(window).height()
            w1 = this.elements.image.width
            h1 = this.elements.image.height

            if (zoom) {
                h2 = height - 2 * margin
                w2 = (h2 / h1) * w1
                w2 = Math.min(w2, width - 2 * margin)
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
                left += (width - w2) * 0.5
                left = left + "px"
                scroll = offset.top - margin
            } else {
                left = ""
                scroll = 0
            }

            $('html, body').animate({scrollTop: scroll});

            this.elements.frame.css({
                "width": w2 + "px",
                "height": h2 + "px",
                "left": left
            })
            this.elements.container.css({
                "height": h2 + "px",
                "width": w2 + "px",
            })

            // Update the assignments, not based on percentages
            for (var index = 0; index < this.entries.length; index++) {
                entry = this.entries[index]
                // We only need to update the assignments for parents, which will update all of their sub-entries
                if(entry !== undefined && entry.parent == null) {
                    this.assignment_entry(index)
                }
            }

            // Update console
            this.refresh()
        }

        BBoxAnnotator.prototype.zoom_start = function() {
            this.state.zoom = true
            this.resize()
        }

        BBoxAnnotator.prototype.zoom_finish = function() {
            this.state.zoom = false
            this.resize()
        }

        BBoxAnnotator.prototype.update_mode = function(proposed_mode) {
            console.log(proposed_mode)
            this.selector_cancel(null)
            this.options.mode = this.bbs.update_mode(proposed_mode)
        }

        BBoxAnnotator.prototype.refresh = function() {
            if (this.options.callbacks.onchange != null) {
                return this.options.callbacks.onchange(this.entries)
            }
        }

        BBoxAnnotator.prototype.delete_entry = function(index, parent_index) {
            var indices, entry, element

            parent_index !== undefined || (parent_index = null)

            if (index == null) {
                return
            }

            // Recursively delete any sub-entries that belong to the parent index
            for (var index_entry = 0; index_entry < this.entries.length; index_entry++) {
                var parent

                parent = this.entries[index_entry].parent
                if (parent != null && parent == index) {
                    index = this.delete_entry(index_entry, index)
                    index_entry = 0
                }
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

            // Now we need to fix the sub-entry parent indices
            for (var index_entry = 0; index_entry < this.entries.length; index_entry++) {
                var parent

                parent = this.entries[index_entry].parent
                if (parent != null) {
                    this.entries[index_entry].parent = indices.indexOf(parent)
                }
            }

            // Refresh the display
            this.refresh()

            // Trigger ondelete
            if (bba.options.callbacks.ondelete != null) {
                return bba.options.callbacks.ondelete(entry[0])
            }

            if (parent_index == null) {
                return null
            } else {
                return indices.indexOf(parent_index)
            }
        }

        BBoxAnnotator.prototype.rotate_element = function(element, angle) {
            var css_rotate

            css_rotate = {
                "transform": "rotate(" + angle + "rad)",
                "msTransform": "rotate(" + angle + "rad)",
                "-o-transform": "rotate(" + angle + "rad)",
                "-ms-transform": "rotate(" + angle + "rad)",
                "-moz-transform": "rotate(" + angle + "rad)",
                "-sand-transform": "rotate(" + angle + "rad)",
                "-webkit-transform": "rotate(" + angle + "rad)",
            }
            element.css(css_rotate)
        }

        BBoxAnnotator.prototype.add_entry = function(entry, notify) {
            var invalid, width, height, element, css_no_select, css_margin, css_handles, configurations, handle, index

            // Check the notify boolean variable
            notify !== undefined || (notify = false)

            // Check the required bounding box location dimensions are specified
            invalid = false
            entry.percent !== undefined || (invalid = true)
            entry.angles !== undefined || (invalid = true)

            if (!invalid) {
                entry.percent.top !== undefined || (invalid = true)
                entry.percent.left !== undefined || (invalid = true)
                entry.percent.width !== undefined || (invalid = true)
                entry.percent.height !== undefined || (invalid = true)
                entry.angles.theta !== undefined || (invalid = true)

                if (!invalid) {
                    width = this.elements.frame.width()
                    height = this.elements.frame.height()

                    // Calculate the final values in percentage space
                    entry.pixels = {}
                    entry.pixels.left = (entry.percent.left / 100.0) * width
                    entry.pixels.top = (entry.percent.top / 100.0) * height
                    entry.pixels.width = (entry.percent.width / 100.0) * width
                    entry.pixels.height = (entry.percent.height / 100.0) * height
                } else {
                    console.log("[BBoxAnnotator] Invalid entry provided: incomplete bounding box dimensions and/or angle")
                    console.log(entry)
                    return
                }
            } else {
                console.log("[BBoxAnnotator] Invalid entry provided: no bounding box dimensions")
                console.log(entry)
                return
            }

            // Ensure the entry has the required optional structures
            entry.label !== undefined || (entry.label = null)
            entry.parent !== undefined || (entry.parent = null)
            if(entry.parent == null && this.state.focus != null) {
                entry.parent = this.state.focus
            }
            entry.metadata !== undefined || (entry.metadata = {})
            entry.highlighted !== undefined || (entry.highlighted = false)
            entry.closest !== undefined || (entry.closest = null)

            console.log('[BBoxAnnotator] Adding entry for:')
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
            if(entry.parent != null) {
                element.bbox.css({
                    // "border-left": this.options.border.width + "px dashed " + this.options.colors.default,
                    // "border-bottom": this.options.border.width + "px dashed " + this.options.colors.default,
                    // "border-right": this.options.border.width + "px dashed " + this.options.colors.default,
                    "outline": this.options.border.width + "px solid " + this.options.colors.subentry,
                })
            }
            // Apply rotation to the bbox
            this.rotate_element(element.bbox, entry.angles.theta)

            this.elements.frame.append(element.bbox)

            // If there is a parent add a line for assigning sub-entry
            if(entry.parent != null) {
                // Create diagonal selector object
                element.assignment = $('<div id="' + this.options.classes.assignment + '"></div>')
                element.assignment.css({
                    "height": 0 + "%",
                    "border-top": "1px solid " + this.options.border.color,
                    "position": "absolute",
                    "transform-origin": "0% 0%",
                })
                this.elements.frame.append(element.assignment)

                if ( ! this.options.subentries.assignments) {
                    element.assignment.hide()
                }
            } else {
                element.assignment = null
            }

            // // Add assignment anchor
            // element.anchor = $('<div id="' + this.options.classes.assignment + '-anchor"></div>')
            // css_margin = Math.floor((this.options.handles.rotate.diameter + 1) * 0.5)
            // element.anchor.css({
            //     "top": 50 + "%",
            //     "left": 50 + "%",
            //     "position": "absolute",
            //     "width": this.options.handles.rotate.diameter + "px",
            //     "height": this.options.handles.rotate.diameter + "px",
            //     "margin-left": "-" + css_margin + "px",
            //     "margin-top": "-" + css_margin + "px",
            //     // "border": "1px solid #333333",
            //     "cursor": "pointer",
            //     "border-radius": this.options.handles.rotate.diameter + "px",
            //     "-moz-border-radius": this.options.handles.rotate.diameter + "px",
            //     "-webkit-border-radius": this.options.handles.rotate.diameter + "px",
            //     "background-color": this.options.colors.default,
            //     "z-index": 999,
            // })
            // element.bbox.append(element.anchor)

            // Add the label to the bbox to denote the ID
            element.label = $('<div class="' + this.options.classes.label + '"></div>')
            element.label.css({
                "overflow": "visible",
                "display": "inline-block",
                "background-color": this.options.colors.default,
                "opacity": "0.8",
                "color": "#333333",
                "padding": "1px 3px",
                "font-size": "12px",
            })
            element.bbox.append(element.label)

            // Add the close button to the bbox
            element.close = $('<div class="' + this.options.classes.close + '"></div>')
            element.close.css({
                "position": "absolute",
                "top": "5px",
                "right": "5px",
                "width": "20px",
                "height": "0",
                "border": "2px solid #FFFFFF",
                "margin-left": "-10px",
                "padding": "16px 0 0 0",
                "color": "#FFFFFF",
                "background-color": "#003300",
                "cursor": "pointer",
                "text-align": "center",
                "overflow": "visible",
                "border-radius": "18px",
                "-moz-border-radius": "18px",
                "-webkit-border-radius": "18px",
            })
            element.bbox.append(element.close)

            // Add the X to the close button, which we do not need to keep track of
            temp = $("<div></div>")
            temp.html("&#215").css({
                "position": "absolute",
                "top": "-2px",
                "left": "0",
                "width": "16px",
                "display": "block",
                "text-align": "center",
                "font-size": "20px",
                "line-height": "20px",
            })
            element.close.append(temp)

            // Hide close button if diabled
            if (!this.options.handles.close.enabled) {
                element.close.hide()
            }

            // Rotation CSS for all handles
            css_margin = Math.floor((this.options.handles.rotate.diameter + 2) * 0.5)
            css_handles = {
                "position": "absolute",
                "width": this.options.handles.rotate.diameter + "px",
                "height": this.options.handles.rotate.diameter + "px",
                "margin-right": "-" + css_margin + "px",
                "margin-top": "-" + css_margin + "px",
                "border": "1px solid #333333",
                "cursor": "pointer",
                "border-radius": this.options.handles.rotate.diameter + "px",
                "-moz-border-radius": this.options.handles.rotate.diameter + "px",
                "-webkit-border-radius": this.options.handles.rotate.diameter + "px",
                "background-color": this.options.colors.default,
            }

            // Add the rotation handle to the bbox
            element.rotate = $('<div class="' + this.options.classes.rotate + '"></div>')
            element.rotate.prop(this.options.props.rotate, "true")
            element.rotate.css(css_handles)
            element.rotate.css({
                "top": "15px",
                "right": "50%",
            })
            if (!this.options.handles.rotate.enabled) {
                element.rotate.hide()
            }
            element.bbox.append(element.rotate)

            // Resize CSS for all handles
            css_margin = Math.floor((this.options.handles.resize.diameter + 2) * 0.5)
            css_handles = {
                "position": "absolute",
                "width": this.options.handles.resize.diameter + "px",
                "height": this.options.handles.resize.diameter + "px",
                "margin-right": "-" + css_margin + "px",
                "margin-top": "-" + css_margin + "px",
                "border": "1px solid #333333",
                "cursor": "pointer",
                "border-radius": this.options.handles.resize.diameter + "px",
                "-moz-border-radius": this.options.handles.resize.diameter + "px",
                "-webkit-border-radius": this.options.handles.resize.diameter + "px",
                "background-color": "#FFFFFF",
            }

            // Add the resize handle to the bbox
            configurations = {
                "nw": {
                    "top": "0px",
                    "left": "-6px",
                },
                "n": {
                    "top": "0px",
                    "right": "50%",
                },
                "ne": {
                    "top": "0px",
                    "right": "0px",
                },
                "e": {
                    "top": "50%",
                    "right": "0px",
                },
                "se": {
                    "right": "0px",
                    "bottom": "-6px",
                },
                "s": {
                    "right": "50%",
                    "bottom": "-6px",
                },
                "sw": {
                    "left": "-6px",
                    "bottom": "-6px",
                },
                "w": {
                    "top": "50%",
                    "left": "-6px",
                },
            }
            element.resize = {}
            for (var key in configurations) {
                handle = $('<div id="' + this.options.ids.resize + '-' + key + '" class="' + this.options.classes.resize + '"></div>')
                handle.prop(this.options.props.resize, key)
                handle.css(css_handles)
                handle.css(configurations[key])
                if (!this.options.handles.resize.enabled) {
                    handle.hide()
                }
                element.bbox.append(handle)
                element.resize[key] = handle
            }

            // Register entry event bindings for this new element
            this.register_entry_event_bindings(element)

            // Add the entry JSON and also the jQuery elements
            this.entries.push(entry)
            this.elements.entries.push(element)

            // Set the label for the entry
            index = this.entries.length - 1
            this.label_entry(index, entry.label)

            if(this.state.focus == null) {
                // Un-hover all entries
                this.hover_entry(null)

                // Set the appropriate visibility
                this.entries_style_visible()
            }

            // Update the assignment line, if needed
            this.assignment_entry(index)

            // notify on adding entry from selection
            if (notify && this.options.callbacks.onadd != null) {
                this.options.callbacks.onadd(entry)
            }

            // Refresh the view now that the new bbox has been added
            this.refresh()

            return index
        }

        BBoxAnnotator.prototype.hover_entry = function(index) {
            var entry

            // Always update the state for the hover2 index
            this.state.hover2 = index

            // Do not update the hover entry if currently selecting
            if (this.state.mode == "selector") {
                return
            }
            // Do not update the hover entry if currently dragging
            if (this.state.mode == "drag") {
                return
            }
            // Do not update the hover entry if currently dragging
            if (this.state.mode == "magnet") {
                return
            }
            // Do not update the hover entry if currently rotating
            if (this.state.mode == "rotate") {
                return
            }
            // Do not update the hover entry if currently resizing
            if (this.state.mode == "resize") {
                return
            }

            // Do not update the hover entry if we have a second focus
            if (this.state.focus2 != null) {
                return
            }

            // Check if we are dealing with a sub-entry
            this.state.subentry = null
            if (index != null && this.state.focus == null) {
                entry = this.entries[index]
                if (entry !== undefined && entry.parent != null) {
                    this.state.subentry = index
                    index = entry.parent
                }
            }

            // Update the state for the hover index
            this.state.hover = index

            if (this.state.focus == null) {
                // Update the style of the correct element
                if (index != null) {
                    this.entry_style_hover(index)

                    // Show all sub-entries
                    this.subentries_style_visible(index)
                } else {
                    for (var index = 0; index < this.elements.entries.length; index++) {
                        this.entry_style_default(index)
                    }

                    // Show all sub-entries
                    this.subentries_style_visible(null)
                }
            }
            else if(this.state.focus != index) {
                if (index != null) {
                    this.entry_style_hover(index)
                } else {
                    for (var index_entry = 0; index_entry < this.elements.entries.length; index_entry++) {
                        entry = this.entries[index_entry]
                        if(entry !== undefined && entry.parent != null && entry.parent == this.state.focus) {
                            this.entry_style_default(index_entry)
                        }
                    }
                }
            }

            // Update the onhover
            if (this.options.callbacks.onhover != null) {
                if(this.state.focus == null) {
                    if (this.state.hover == null) {
                        entry = null
                    }
                    else {
                        entry = this.entries[this.state.hover]
                    }
                    return this.options.callbacks.onhover(this.state.hover, entry)
                } else {
                    if(this.state.hover == null) {
                        entry = this.entries[this.state.focus]
                        return this.options.callbacks.onhover(this.state.focus, entry)
                    } else {
                        entry = this.entries[this.state.hover]
                        return this.options.callbacks.onhover(this.state.hover, entry)
                    }
                }
            }
        }

        BBoxAnnotator.prototype.focus_entry = function(index) {
            var entry

            if (index == null) {
                return
            }

            // Update the style of the correct element
            if (this.state.focus == null) {
                // Update the state for the hover index
                this.state.focus = index
                this.state.focus2 = null
                entry = this.entries[index]

                // Set the style of the focused entry
                this.entry_style_focus(index)

                // Hide all the rest of the bounding boxes
                this.entries_style_hidden(index)

                // Show the required sub-entries as well
                this.subentries_style_visible(index)

                // If there is a subentry, hover it when focusing
                if(this.state.subentry != null) {
                    this.hover_entry(this.state.subentry)
                }
            } else {
                // Second focus
                if (index != this.state.focus) {
                    if (this.state.focus2 == null) {
                        var css_transparent

                        this.state.focus2 = index

                        // Set the style of the focused entry
                        this.entry_style_focus(index)

                        // Make the other sub-entries semi-transparent
                        css_transparent = {
                            'opacity': this.options.colors.transparent,
                        }
                        for (var index_entry = 0; index_entry < this.elements.entries.length; index_entry++) {
                            var entry, element

                            entry = this.entries[index_entry]
                            if (entry.parent == this.state.focus) {
                                if (this.state.focus2 != index_entry) {
                                    element = this.elements.entries[index_entry]
                                    element.bbox.css(css_transparent)
                                    if (element.assignment != null) {
                                        element.assignment.css(css_transparent)
                                    }
                                }
                            }
                        }

                        // Update the onfocus
                        if (this.options.callbacks.onfocus != null) {
                            if (this.state.focus2 == null) {
                                entry = null
                            } else {
                                entry = this.entries[this.state.focus2]
                            }
                            return this.options.callbacks.onfocus(this.state.focus2, entry)
                        }
                    } else {
                        if(this.state.hover2 == this.state.focus2) {
                            // Set the style of the focused entry
                            this.entry_style_hover(this.state.focus2)
                        } else {
                            this.entry_style_default(this.state.focus2)
                        }

                        this.state.focus2 = null
                        this.state.hover = this.state.hover2

                        // Make the other sub-entries semi-transparent
                        css_transparent = {
                            'opacity': 1.0,
                        }
                        for (var index_entry = 0; index_entry < this.elements.entries.length; index_entry++) {
                            var entry, element

                            entry = this.entries[index_entry]
                            if (entry.parent == this.state.focus) {
                                element = this.elements.entries[index_entry]
                                element.bbox.css(css_transparent)
                                if (element.assignment != null) {
                                    element.assignment.css(css_transparent)
                                }
                            }
                        }

                        // Set the style of the un-focused entry back to hover
                        this.hover_entry(this.state.hover)
                    }
                } else {
                    enabled = true
                    if (this.options.actions.entry.defocus == false) {
                        enabled = false
                    }

                    if (enabled) {
                        // Un-hover all of the sub-entries
                        for (var index_entry = 0; index_entry < this.elements.entries.length; index_entry++) {
                            entry = this.entries[index_entry]
                            if(entry.parent != null && entry.parent == this.state.focus) {
                                this.entry_style_default(index_entry)
                            }
                        }

                        // Unset the state of the hover index
                        this.state.focus = null

                        // Set the style of the un-focused entry back to hover
                        this.hover_entry(this.state.hover)

                        // Un-hide all the rest of the bounding boxes
                        this.entries_style_visible(index)

                        // Show the required sub-entries as well
                        this.subentries_style_visible(index)
                    }
                }
            }

            // Update the onfocus
            if (this.options.callbacks.onfocus != null) {
                if (this.state.focus == null) {
                    entry = null
                } else {
                    entry = this.entries[this.state.focus]
                }
                return this.options.callbacks.onfocus(this.state.focus, entry)
            }
        }

        BBoxAnnotator.prototype.anchor_update = function(event, handle) {
            var index, element, offset, position, position_resize

            index = this.state.hover

            if (index == null) {
                return
            }

            entry = this.entries[index]
            element = this.elements.entries[index]
            offset = this.elements.container.offset()

            // We need to un-rotate the box before we can get a correct position
            this.rotate_element(element.bbox, 0.0)

            // Get the accurate position from jQuery
            position = element.bbox.position()

            // Update the cursor and element anchors
            this.state.anchors = {}
            this.state.anchors.cursor = {
                x: event.pageX - offset.left,
                y: event.pageY - offset.top,
            }
            this.state.anchors.element = {
                x: position.left,
                y: position.top,
                width: element.bbox.width() + this.options.border.width * 2.0,
                height: element.bbox.height() + this.options.border.width * 2.0,
                theta: entry.angles.theta,
                right: entry.angles.theta - Math.PI * 0.5,
                handle: handle,
            }

            // Re-rotate the box now that we have accurate coordinates
            this.rotate_element(element.bbox, entry.angles.theta)

            // Get resize handles locations
            position = element.bbox.position()
            this.state.anchors.resize = {}
            for (var key in element.resize) {
                position_resize = element.resize[key].position()
                this.state.anchors.resize[key] = {
                    x: position.left + position_resize.left,
                    y: position.top + position_resize.top,
                }
            }
        }

        BBoxAnnotator.prototype.label_entry = function(index, label) {
            var entry, element, holder, font

            entry = this.entries[index]
            element = this.elements.entries[index]

            // Set the label in the entry's metadata
            entry.label = label

            // Update the element's text box with the new label
            holder = null
            if(entry.parent == null && entry.label != null) {
                holder = entry.label
                font = 12
            } else if(entry.parent != null && entry.metadata.type != null && this.options.subentries.label) {
                if(entry.metadata.type != '____') {
                    holder = entry.metadata.type
                    font = 9
                }
            }

            element.label.css({
                "font-size": font + "px",
            })

            // Update label in HTML
            if (holder != null || entry.highlighted) {
                if(holder == null) {
                    holder = ''
                }
                if(entry.highlighted) {
                    // var species_strs = []
                    // species_strs['giraffe_masai']       = 'Masai Giraffe'
                    // species_strs['giraffe_reticulated'] = 'Reticulated Giraffe'
                    // species_strs['turtle_sea']          = 'Sea Turtle'
                    // species_strs['turtle_hawksbill']    = 'Sea Turtle'
                    // species_strs['turtle_green']        = 'Sea Turtle'
                    // species_strs['whale_fluke']         = 'Whale Fluke'
                    // species_strs['zebra_grevys']        = 'Grevy\'s Zebra'
                    // species_strs['zebra_plains']        = 'Plains Zebra'

                    // value1 = entry['metadata']['viewpoint1']
                    // value2 = entry['metadata']['viewpoint2']
                    // value3 = entry['metadata']['viewpoint3']
                    // species = entry['metadata']['species']
                    // tag = normalize_viewpoint(value1, value2, value3)
                    // holder = species_strs[species] + ' (' + tag + ')'
                    // element.label.html(holder)

                    // holder = holder + "*"
                    // element.label.html(holder)

                    holder = "*"
                    element.label.html(holder)
                } else {
                    holder = ''
                }
                element.label.html(holder)
                element.label.show()
            } else {
                element.label.hide()
            }
        }

        BBoxAnnotator.prototype.get_metadata = function() {
            var index, entry

            index = this.state.hover
                // Do not update if there is not a hover entry
            if (index == null) {
                // Try again by checking if there is a focus
                index = this.state.focus
                if (index == null) {
                    return null
                }
            }

            entry = this.entries[index]
            return entry.metadata
        }

        BBoxAnnotator.prototype.set_metadata = function(metadata) {
            var index, entry

            index = this.state.hover
                // Do not update if there is not a hover entry
            if (index == null) {
                index = this.state.focus
                if (index == null) {
                    return null
                }
            }

            entry = this.entries[index]
            entry.metadata = metadata

            this.label_entry(index, entry.label)

            this.refresh()
        }

        BBoxAnnotator.prototype.set_highlighted = function(highlighted) {
            var index, entry

            index = this.state.hover
                // Do not update if there is not a hover entry
            if (index == null) {
                index = this.state.focus
                if (index == null) {
                    return null
                }
            }

            entry = this.entries[index]
            entry.highlighted = highlighted

            this.label_entry(index, entry.label)

            this.refresh()
        }

        BBoxAnnotator.prototype.anchor_entry = function(event) {
            var index, entry, element, closest_dist, closest_key
            var handle, offset, diff_x, diff_y, dist

            index = this.state.hover
            entry = this.entries[index]
            element = this.elements.entries[index]

            // Do not update if there is a focused entry
            if (this.state.focus != null) {
                if (entry.parent != null && entry.parent != this.state.focus) {
                    return
                }
            }

            if (entry === undefined || element === undefined) {
                return
            }

            // Get the furthest point
            closest_dist = Infinity
            closest_key = null
            for (var key in element.resize) {
                handle = element.resize[key]
                offset = handle.offset()
                diff_x = event.pageX - offset.left
                diff_y = event.pageY - offset.top
                dist = Math.sqrt(Math.pow(diff_x, 2) + Math.pow(diff_y, 2))
                // Get the closest
                if (dist < closest_dist) {
                    closest_dist = dist
                    closest_key = key
                }
                // Set all anchors to be the standard
                handle.css({
                    "background-color": "#FFFFFF",
                })
            }

            // if (closest_key == "nw") {
            //     closest_anchor = "se"
            // } else if (closest_key == "n") {
            //     closest_anchor = "s"
            // } else if (closest_key == "ne") {
            //     closest_anchor = "sw"
            // } else if (closest_key == "e") {
            //     closest_anchor = "w"
            // } else if (closest_key == "se") {
            //     closest_anchor = "nw"
            // } else if (closest_key == "s") {
            //     closest_anchor = "n"
            // } else if (closest_key == "sw") {
            //     closest_anchor = "ne"
            // } else if (closest_key == "w") {
            //     closest_anchor = "e"
            // }
            // handle = element.resize[closest_anchor]

            entry['closest'] = closest_key
            handle = element.resize[closest_key]
            handle.css({
                "background-color": this.options.colors.anchor,
            })
        }

         BBoxAnnotator.prototype.assignment_entry = function(index) {
            var index, entry, element, parent, width, height, centers, delta, hypo, angle

            // Do not update if there is not a hover entry
            if (index == null) {
                return
            }

            entry = this.entries[index]
            element = this.elements.entries[index]

            // Check to make sure this entry has a parent, if not move all sub-entries
            if (entry.parent == null) {
                for (var index_entry = 0; index_entry < this.entries.length; index_entry++) {
                    entry = this.entries[index_entry]
                    if(entry.parent != null && entry.parent == index) {
                        this.assignment_entry(index_entry)
                    }
                }
            }

            // Sanity check if there is no assignment element (i.e. not a subentry)
            if (element.assignment == null) {
                return
            }

            // Get the parent
            parent = this.entries[entry.parent]

            width = this.elements.frame.width()
            height = this.elements.frame.height()

            // Compute centers
            centers = {}
            centers.entry = {
                x: (entry.percent.left + entry.percent.width * 0.5) / 100.0 * width,
                y: (entry.percent.top + entry.percent.height * 0.5) / 100.0 * height,
            }
            centers.parent = {
                x: (parent.percent.left + parent.percent.width * 0.5) / 100.0 * width,
                y: (parent.percent.top + parent.percent.height * 0.5) / 100.0 * height,
            }

            delta = {
                x: centers.entry.x - centers.parent.x,
                y: centers.entry.y - centers.parent.y,
            }

            hypo = Math.sqrt(Math.pow(delta.x, 2) + Math.pow(delta.y, 2))
            angle = calculate_angle(centers.entry, centers.parent)

            element.assignment.css({
                "top": (entry.percent.top + entry.percent.height * 0.5) + "%",
                "left": (entry.percent.left + entry.percent.width * 0.5) + "%",
                "width": hypo + "px",
            })
            this.rotate_element(element.assignment, angle)
        }

        BBoxAnnotator.prototype.move_entry = function(event) {
            var index, entry, element, offset, width, height

            index = this.state.hover

            // Do not update if there is not a hover entry
            if (index == null) {
                return
            }

            if (this.state.focus2 != null) {
                return
            }

            // Do not update if there is a focused entry
            if (this.state.focus != null) {
                entry = this.entries[index]
                if(entry.parent != this.state.focus) {
                    return
                }
            }

            entry = this.entries[index]
            element = this.elements.entries[index]

            if (entry === undefined || element === undefined) {
                return
            }

            var enabled = true;
            if (entry.parent == null) {
                if (this.options.actions.entry.translation == false) {
                    enabled = false;
                }
            } else {
                if (this.options.actions.subentry.translation == false) {
                    enabled = false;
                }
            }

            if(enabled) {
                offset = this.elements.container.offset()

                offset = {
                    x: event.pageX - offset.left - this.state.anchors.cursor.x,
                    y: event.pageY - offset.top - this.state.anchors.cursor.y,
                }

                entry.pixels.left = this.state.anchors.element.x + offset.x,
                entry.pixels.top = this.state.anchors.element.y + offset.y,

                width = this.elements.frame.width()
                height = this.elements.frame.height()

                // Calculate the final values in percentage space
                entry.percent.left = 100.0 * entry.pixels.left / width
                entry.percent.top = 100.0 * entry.pixels.top / height

                // // If the shift is held, move parts as well
                // if(event.shiftKey) {
                // }

                element.bbox.css({
                    "top": entry.percent.top + "%",
                    "left": entry.percent.left + "%",
                })
            }

            // Update the assignments, if needed
            this.assignment_entry(index)

            // Update the entries
            this.refresh()
        }

        BBoxAnnotator.prototype.move_entry_correction = function(event, correction) {
            var index, element, position, width, height

            index = this.state.hover

            if (index == null) {
                return
            }

            if (this.state.focus2 != null) {
                return
            }

            // If the event is valid (meaning we have an active, hovered bbox) don't scroll the screen
            event.preventDefault()

            entry = this.entries[index]
            element = this.elements.entries[index]

            // We need to un-rotate the box before we can get a correct position
            this.rotate_element(element.bbox, 0.0)
                // Get the accurate position from jQuery
            position = element.bbox.position()
                // Re-rotate the box now that we have accurate coordinates
            this.rotate_element(element.bbox, entry.angles.theta)

            entry.pixels.left = position.left + correction.x
            entry.pixels.top = position.top + correction.y

            width = this.elements.frame.width()
            height = this.elements.frame.height()

            // Calculate the final values in percentage space
            entry.percent.left = 100.0 * entry.pixels.left / width
            entry.percent.top = 100.0 * entry.pixels.top / height

            // // If the shift is held, move parts as well
            // if(event.shiftKey) {
            // }

            element.bbox.css({
                "top": entry.percent.top + "%",
                "left": entry.percent.left + "%",
            })

            // Update the assignments, if needed
            this.assignment_entry(index)

            // Update the entries
            this.refresh()
        }

        BBoxAnnotator.prototype.rotate_entry = function(event) {
            var index, entry, element, offset, center, cursor, angles, delta, correction

            index = this.state.hover
                // Do not update if there is not a hover entry
            if (index == null) {
                return
            }

            if (this.state.focus2 != null) {
                return
            }

            entry = this.entries[index]
            element = this.elements.entries[index]
            offset = this.elements.container.offset()

            var enabled = true;
            if (entry.parent == null) {
                if (this.options.actions.entry.rotation == false) {
                    enabled = false;
                }
            } else {
                if (this.options.actions.subentry.rotation == false) {
                    enabled = false;
                }
            }

            if (enabled) {
                center = {
                    x: this.state.anchors.element.x + this.state.anchors.element.width * 0.5,
                    y: this.state.anchors.element.y + this.state.anchors.element.height * 0.5,
                }
                cursor = {
                    x: event.pageX - offset.left,
                    y: event.pageY - offset.top,
                }

                angles = {}
                angles.anchor = calculate_angle(center, this.state.anchors.cursor)
                angles.cursor = calculate_angle(center, cursor)

                delta = angles.cursor - angles.anchor
                correction = this.state.anchors.element.theta + delta

                // If the Shift key is pressed, block to steps
                if (event.shiftKey) {
                    correction /= this.options.steps.rotate
                    correction = Math.round(correction)
                    correction *= this.options.steps.rotate
                }

                entry.angles.theta = correction
                this.rotate_element(element.bbox, entry.angles.theta)
            }

            // Update the assignments, if needed
            this.assignment_entry(index)

            // Update the entries
            this.refresh()
        }

        BBoxAnnotator.prototype.orient_entry = function(event, direction, index) {
            var index, entry, element, holder, width, height

            index !== undefined || (index = this.state.hover)

                // Do not update if there is not a hover entry
            if (index == null) {
                return
            }

            if (direction != "left" && direction != "right") {
                return
            }

            entry = this.entries[index]
            element = this.elements.entries[index]

            holder = JSON.parse(JSON.stringify(entry.percent))

            width = this.elements.frame.width()
            height = this.elements.frame.height()

            center = {
                x: holder.left + (holder.width * 0.5),
                y: holder.top + (holder.height * 0.5),
            }

            entry.percent.width = (holder.height * height) / width
            entry.percent.height = (holder.width * width) / height
            entry.percent.left = center.x - (entry.percent.width * 0.5)
            entry.percent.top = center.y - (entry.percent.height * 0.5)

            element.bbox.css({
                "top": entry.percent.top + "%",
                "left": entry.percent.left + "%",
                "width": entry.percent.width + "%",
                "height": entry.percent.height + "%",
            })

            if (direction == "left") {
                entry.angles.theta -= Math.PI * 0.5
            } else {
                entry.angles.theta += Math.PI * 0.5
            }
            this.rotate_element(element.bbox, entry.angles.theta)

            // Update the entries
            this.refresh()
        }

        BBoxAnnotator.prototype.background_entry = function(event, index, subentry) {
            var index, indices, entry, element, holder

            subentry !== undefined || (subentry = true)

            // Do not update if there is not an index
            if (index == null) {
                return
            }

            // If we are a parent with associated subentries, background the subentries first
            entry = this.entries[index]
            if (entry.parent == null) {
                // Disallow background entry if in focus and a parent
                if(this.state.focus != null) {
                    return
                }
                // Background all of the sub-entries related to this parent first
                for (var index_entry = 0; index_entry < this.entries.length; index_entry++) {
                    var parent

                    parent = this.entries[index_entry].parent
                    if(parent != null && parent == index) {
                        this.background_entry(entry, index_entry, false)
                        index += 1
                    }
                }
            } else if (subentry) {
                // Disallow background entry if in focus2 and a subentry
                if(this.state.focus == null) {
                    return
                }

                if(this.state.focus2 != null) {
                    return
                }

                // We want to take all of the subentries in this entry and rotate
                // the elements that are lower in index than the currently hovered
                // index.  Since all parent indices are not changing, we can do this
                // as a drop-in action.

                // Background all of the sub-entries related to this parent first
                indices = []
                for (var index_entry = 0; index_entry < this.entries.length; index_entry++) {
                    var parent

                    parent = this.entries[index_entry].parent
                    if(parent != null && parent == this.state.focus) {
                        indices.push(index_entry)
                        // We want to get only the subentries before the indexed subentry
                        if(index_entry == index) {
                            break
                        }
                    }
                }

                // Save first index
                if(indices.length > 1) {
                    // Get indices of last two items
                    index_last = indices[indices.length - 1]
                    index_penultimate = indices[indices.length - 2]

                    // Get last item
                    entry_last = this.entries[index_last]
                    element_last = this.elements.entries[index_last]

                    entry_penultimate = this.entries[index_penultimate]
                    element_penultimate = this.elements.entries[index_penultimate]

                    // Swap last two items in entries and elements
                    this.entries[index_last] = entry_penultimate
                    this.elements.entries[index_last] = element_penultimate
                    this.entries[index_penultimate] = entry_last
                    this.elements.entries[index_penultimate] = element_last

                    // Swap last two DOM elements
                    element_last.bbox.swapWith(element_penultimate.bbox)

                    // Continue, recursively
                    this.background_entry(event, index_penultimate)
                }

                // Do not continue, simply return
                return
            }

            // Keep track of the current indices
            indices = []
            for (var counter = 0; counter < this.entries.length; counter++) {
                indices.push(counter)
            }

            // Move the entry to the start of the list
            entry = this.entries.splice(index, 1)
            this.entries.unshift(entry[0])

            // Perform the exact same operation on the indices
            entry = indices.splice(index, 1)
            indices.unshift(entry[0])

            // Move the element to the start of the list
            element = this.elements.entries.splice(index, 1)
            this.elements.entries.unshift(element[0])

            // Prepend the element to the frame
            holder = element[0].bbox.detach()
            this.elements.frame.prepend(holder)

            // Now we need to fix the sub-entry parent indices
            for (var index = 0; index < this.entries.length; index++) {
                var parent

                parent = this.entries[index].parent
                if (parent != null) {
                    this.entries[index].parent = indices.indexOf(parent)
                }
            }

            // Update the entries
            this.refresh()
        }

        BBoxAnnotator.prototype.resize_start = function(resize_element, event) {
            var index, entry, element, handle, border_side

            // Find the parent bbox of the close button and the bbox's index
            parent = resize_element.parent("." + bba.options.classes.bbox)
            index = parent.prevAll("." + bba.options.classes.bbox).length
            entry = bba.entries[index]
            element = bba.elements.entries[index]
            handle = resize_element.prop(bba.options.props.resize)

            // Set the other entries to be transparent
            bba.entries_style_transparent(bba.options.colors.transparent, index)

            if (entry.parent != null) {
                element2 = bba.elements.entries[entry.parent]
                element2.bbox.css({
                    'opacity': 1.0,
                })
            }

            // Get the orientation
            bba.state.orientation = Math.floor(entry.angles.theta / (Math.PI * 0.5))
            while (bba.state.orientation < 0) {
                bba.state.orientation += 4
            }
            while (bba.state.orientation > 4) {
                bba.state.orientation -= 4
            }
            bba.state.orientation = Math.floor(bba.state.orientation)

            // Set the border-top based on the correct orientation
            if(bba.state.orientation == 0) {
                border_side = null
            } else if(bba.state.orientation == 1) {
                border_side = "border-right"
            } else if(bba.state.orientation == 2) {
                border_side = "border-bottom"
            } else if(bba.state.orientation == 3) {
                border_side = "border-left"
            }

            if(border_side != null) {
                // Set the element border with the correct orientation
                element.bbox.css("border-top", bba.options.border.width + "px solid" + bba.options.colors.hover)
                element.bbox.css(border_side, (bba.options.border.width * 1.5) + "px dotted " + bba.options.colors.hover)
            }

            // Fix the handle based on the orientation
            for (var index = bba.state.orientation; index > 0; index--) {
                bba.orient_entry(event, "left")

                // Update the handle
                if (handle == "nw") {
                    handle = "ne"
                } else if (handle == "n") {
                    handle = "e"
                } else if (handle == "ne") {
                    handle = "se"
                } else if (handle == "e") {
                    handle = "s"
                } else if (handle == "se") {
                    handle = "sw"
                } else if (handle == "s") {
                    handle = "w"
                } else if (handle == "sw") {
                    handle = "nw"
                } else if (handle == "w") {
                    handle = "n"
                }
            }

            // Hide the close and rotate buttons
            if (!bba.options.debug) {
                element.label.hide()
                element.close.hide()
                element.rotate.hide()
            }

            // Dis-allow selection by drag selection
            document.onselectstart = function() {
                return false
            }
            bba.anchor_update(event, handle)
        }

        BBoxAnnotator.prototype.resize_finish = function(event) {
            var element, index

            // Correct the orientation back to the original state
            for (var index = bba.state.orientation; index > 0; index--) {
                bba.orient_entry(event, "right")
            }

            // Set the other entries to be transparent
            bba.entries_style_transparent(1.0)

            if (bba.state.hover != null) {
                entry = bba.entries[bba.state.hover]
                element = bba.elements.entries[bba.state.hover]

                // Set the element border with the correct orientation
                element.bbox.css({
                    "border": bba.options.border.width + "px solid " + bba.options.colors.hover,
                    "border-top": (bba.options.border.width * 1.5) + "px dotted " + bba.options.colors.hover,
                })

                bba.label_entry(bba.state.hover, entry.label)
                if (bba.options.handles.close.enabled) {
                    element.close.show()
                }
                if (bba.options.handles.rotate.enabled) {
                    element.rotate.show()
                }
            }

            document.onselectstart = null
            bba.state.mode = "free"
        }

        BBoxAnnotator.prototype.resize_entry = function(event) {
            var index, handle, entry, element, anchor
            var offset, cursor, delta, theta, hypo, ratio
            var angle, diff_angle, diff, components, offset
            var width, height, position, position_resize, temp, correction

            index = this.state.hover
            handle = this.state.anchors.element.handle

            // Do not update if there is not a hover entry
            if (index == null || handle == null) {
                return
            }

            entry = this.entries[index]
            element = this.elements.entries[index]
            offset = this.elements.container.offset()

            cursor = {
                x: event.pageX - offset.left,
                y: event.pageY - offset.top,
            }

            delta = {
                x: cursor.x - this.state.anchors.cursor.x,
                y: cursor.y - this.state.anchors.cursor.y,
            }

            theta = this.state.anchors.element.theta
            hypo = Math.sqrt(Math.pow(delta.x, 2) + Math.pow(delta.y, 2))
            angle = calculate_angle(this.state.anchors.cursor, cursor)
            diff_angle = angle - theta

            diff = {
                x: Math.cos(diff_angle) * hypo,
                y: Math.sin(diff_angle) * hypo,
            }

            // Calculate components for the in-axis x and y offsets
            hypo = 0.0
            components = {
                x: {},
                y: {},
            }
            components.x = {
                x: Math.cos(theta) * diff.x,
                y: Math.sin(theta) * diff.x,
            }
            components.x.hypo = Math.sqrt(Math.pow(components.x.x, 2) + Math.pow(components.x.y, 2))
            if (components.x.x > 0) {
                components.x.hypo *= -1.0
            }
            components.y = {
                x: Math.sin(-theta) * diff.y,
                y: Math.cos(-theta) * diff.y,
            }
            components.y.hypo = Math.sqrt(Math.pow(components.y.x, 2) + Math.pow(components.y.y, 2))
            if (components.y.y > 0) {
                components.y.hypo *= -1.0
            }

            // if(event.shiftKey) {
                // ratio = this.state.anchors.element.width / this.state.anchors.element.height
                // console.log(ratio)
                // components.x.hypo = components.y.hypo * ratio
                // if(components.y.hypo >= components.x.hypo) {
                // } else  {
                //     ratio = this.state.anchors.element.width / this.state.anchors.element.height
                //     components.y.hypo = components.x.hypo / ratio
                // }
            // }

            // Calculate the offset for each handle orientation
            if (handle == "nw") {
                anchor = "se"
                offset = {
                    x: components.x.x,
                    y: components.y.y,
                    width: components.x.hypo,
                    height: components.y.hypo,
                }
            } else if (handle == "n") {
                anchor = "s"
                offset = {
                    x: 0.0,
                    y: components.y.y,
                    width: 0.0,
                    height: components.y.hypo,
                }
            } else if (handle == "ne") {
                anchor = "sw"
                offset = {
                    x: 0.0,
                    y: components.y.y,
                    width: -1.0 * components.x.hypo,
                    height: components.y.hypo,
                }
            } else if (handle == "e") {
                anchor = "w"
                offset = {
                    x: 0.0,
                    y: 0.0,
                    width: -1.0 * components.x.hypo,
                    height: 0.0,
                }
            } else if (handle == "se") {
                anchor = "nw"
                offset = {
                    x: 0.0,
                    y: 0.0,
                    width: -1.0 * components.x.hypo,
                    height: -1.0 * components.y.hypo,
                }
            } else if (handle == "s") {
                anchor = "n"
                offset = {
                    x: 0.0,
                    y: 0.0,
                    width: 0.0,
                    height: -1.0 * components.y.hypo,
                }
            } else if (handle == "sw") {
                anchor = "ne"
                offset = {
                    x: components.x.x,
                    y: 0.0,
                    width: components.x.hypo,
                    height: -1.0 * components.y.hypo,
                }
            } else if (handle == "w") {
                anchor = "e"
                offset = {
                    x: components.x.x,
                    y: 0.0,
                    width: components.x.hypo,
                    height: 0.0,
                }
            }

            // Set all anchors to be the standard (white)
            for (var key in element.resize) {
                element.resize[key].css({
                    "background-color": "#FFFFFF",
                })
            }
            element.resize[handle].css({
                "background-color": this.options.colors.anchor,
            })

            // Temporarily put the width and height to check for bounds check
            top = this.state.anchors.element.y + offset.y
            left = this.state.anchors.element.x + offset.x
            width = this.state.anchors.element.width + offset.width
            height = this.state.anchors.element.height + offset.height

            // If the width and height of the new location is negative
            if (width < 0 || height < 0) {
                return
            }

            var enabled = true;
            if (entry.parent == null) {
                if (this.options.actions.entry.rotation == false) {
                    enabled = false;
                }
            } else {
                if (this.options.actions.subentry.rotation == false) {
                    enabled = false;
                }
            }

            if (enabled) {
                // entry.pixels.top = top
                // entry.pixels.left = left
                entry.pixels.width = width
                entry.pixels.height = height

                width = this.elements.frame.width()
                height = this.elements.frame.height()

                // Calculate the final values in percentage space
                entry.percent.top = 100.0 * entry.pixels.top / height
                entry.percent.left = 100.0 * entry.pixels.left / width
                entry.percent.width = 100.0 * entry.pixels.width / width
                entry.percent.height = 100.0 * entry.pixels.height / height

                element.bbox.css({
                    "top": entry.percent.top + "%",
                    "left": entry.percent.left + "%",
                    "width": entry.percent.width + "%",
                    "height": entry.percent.height + "%",
                })

                ////////////////////////////////////////////////////////////////////////////////////

                // Calculate drift offset and correction
                position = element.bbox.position()
                position_resize = element.resize[anchor].position()

                temp = {
                        x: position.left + position_resize.left,
                        y: position.top + position_resize.top,
                    }
                    // Get current location
                correction = {
                    x: temp.x - this.state.anchors.resize[anchor].x,
                    y: temp.y - this.state.anchors.resize[anchor].y,
                }

                entry.pixels.left -= correction.x
                entry.pixels.top -= correction.y

                // Calculate the final values in percentage space
                entry.percent.left = 100.0 * entry.pixels.left / width
                entry.percent.top = 100.0 * entry.pixels.top / height

                element.bbox.css({
                    "top": entry.percent.top + "%",
                    "left": entry.percent.left + "%",
                })
            }

            ////////////////////////////////////////////////////////////////////////////////////

            if (this.debug.length > 0) {
                this.debug[0].css({
                    "top": this.state.anchors.resize[anchor].y + "px",
                    "left": this.state.anchors.resize[anchor].x + "px",
                })
                this.debug[1].css({
                    "top": temp.y + "px",
                    "left": temp.x + "px",
                })
                this.debug[2].css({
                    "top": (this.state.anchors.cursor.y + components.x.y) + "px",
                    "left": (this.state.anchors.cursor.x + components.x.x) + "px",
                })
                this.debug[3].css({
                    "top": (this.state.anchors.cursor.y + components.y.y) + "px",
                    "left": (this.state.anchors.cursor.x + components.y.x) + "px",
                })
            }

            // Update the assignments, if needed
            this.assignment_entry(index)

            // Update the entries
            this.refresh()
        }

        BBoxAnnotator.prototype.selector_start = function(event) {
            // Check configuration
            if (this.state.focus == null && this.options.actions.entry.addition == false) {
                return
            }
            if (this.state.focus != null && this.options.actions.subentry.addition == false) {
                return
            }

            // Remove any current hover styles
            this.hover_entry(null)

            // Make the other entries semi-transparent
            this.entries_style_transparent(this.options.colors.transparent)

            // If we have a parent, do not hide
            if(this.state.focus != null) {
                element = this.elements.entries[this.state.focus]
                element.bbox.css({
                    'opacity': 1.0,
                })
            }

            // Start the BBoxSelector
            subentry = this.state.focus != null
            this.bbs.start(event, subentry)
            this.state.mode = "selector"

            // notify on adding entry from selection
            if (this.options.callbacks.onselector != null) {
                return this.options.callbacks.onselector()
            }
        }

        BBoxAnnotator.prototype.selector_update = function(event) {
            // Update the BBoxSelector
            this.bbs.update_cursor(event)
        }

        BBoxAnnotator.prototype.selector_stage = function(event) {
            var finish

            // Stage the BBoxSelector
            finish = this.bbs.stage(event)

            if (finish) {
                this.selector_finish(event)
            }
        }

        BBoxAnnotator.prototype.selector_cancel = function(event) {
            if (this.state.mode == "selector") {
                // Get the entry data by finishing the BBoxSelector
                this.bbs.finish(event)

                // Release the mode to "free"
                this.state.mode = "free"

                // Make the other entries semi-transparent
                this.entries_style_transparent(1.0)
            }

            // // Hover if there is an active hover
            // this.hover_entry(this.state.hover2)
        }

        BBoxAnnotator.prototype.selector_finish = function(event) {
            var data, invalid, index

            // If we are in the "selector" mode, finish the BBoxSelector and add the entry
            if (this.state.mode == "selector") {
                // Get the entry data by finishing the BBoxSelector
                data = this.bbs.finish(event)

                invalid = false
                invalid = invalid || (isNaN(data.pixels.left))
                invalid = invalid || (isNaN(data.pixels.top))
                invalid = invalid || (isNaN(data.pixels.width))
                invalid = invalid || (isNaN(data.pixels.height))
                invalid = invalid || (data.pixels.width * data.pixels.height <= this.options.limits.entry.area)
                invalid = invalid || (data.pixels.width <= this.options.limits.entry.width)
                invalid = invalid || (data.pixels.height <= this.options.limits.entry.height)

                // Add the returned entry data reported by the BBoxSelector as a new entry
                if( ! invalid) {
                    index = this.add_entry(data, true)

                    if (this.options.mode == "diagonal2") {
                        if(data.angles.fixed) {
                            bba.orient_entry(event, "right", index)
                        } else {
                            bba.orient_entry(event, "left", index)
                        }
                    }
                }

                // Release the mode to "free"
                this.state.mode = "free"

                // Make the other entries semi-transparent
                this.entries_style_transparent(1.0)
            }

            // // Hover if there is an active hover
            // this.hover_entry(this.state.hover2)
        }

        BBoxAnnotator.prototype.entries_style_transparent = function(opacity, skip) {
            var css_transparent

            // Check the skip index
            skip !== undefined || (skip = null)

            css_transparent = {
                'opacity': opacity,
            }
            // Make the other entries semi-transparent
            for (var index = 0; index < this.elements.entries.length; index++) {
                var element
                if (index != skip) {
                    element = this.elements.entries[index]

                    element.bbox.css(css_transparent)
                    if (element.assignment != null) {
                        element.assignment.css(css_transparent)
                    }
                }
            }
        }

        BBoxAnnotator.prototype.update_subentries_assignments = function(assignments) {
            this.options.subentries.assignments = assignments
            this.update_subentries_options()
        }

        BBoxAnnotator.prototype.update_subentries_visible = function(visible) {
            this.options.subentries.visible = visible
            this.update_subentries_options()
        }

        BBoxAnnotator.prototype.update_subentries_options = function() {
            if(this.state.focus == null) {
                this.entries_style_visible()
                if ( ! this.options.subentries.visible && this.state.hover != null) {
                    this.subentries_style_visible(this.state.hover)
                }
            }
            else {
                // Make the other entries semi-transparent
                for (var index = 0; index < this.elements.entries.length; index++) {
                    entry = this.entries[index]
                    if (entry.parent == this.state.focus) {
                        element = this.elements.entries[index]
                        // Check if we are an entry or sub-entry
                        if (this.options.subentries.visible == false) {
                            element.bbox.hide()
                            // Set the assignment
                            if (element.assignment != null) {
                                element.assignment.hide()
                            }
                        } else {
                            element.bbox.show()
                            // Set the assignment
                            if (element.assignment != null) {
                                if(this.options.subentries.assignments) {
                                    element.assignment.show()
                                } else {
                                    element.assignment.hide()
                                }
                            }
                        }
                    }
                }
            }
        }

        BBoxAnnotator.prototype.entries_style_visible = function(skip) {
            var entry, element

            // Establish the desired parent, if wanted
            skip !== undefined || (skip = null)

            // Make the other entries semi-transparent
            for (var index = 0; index < this.elements.entries.length; index++) {
                if (index != skip) {
                    entry = this.entries[index]
                    element = this.elements.entries[index]
                    // Check if we are an entry or sub-entry
                    if (entry.parent == null || this.options.subentries.visible == true) {
                        element.bbox.show()
                        // Set the assignment
                        if (element.assignment != null) {
                            if(this.options.subentries.assignments) {
                                element.assignment.show()
                            } else {
                                element.assignment.hide()
                            }
                        }
                    } else {
                        element.bbox.hide()
                        // Set the assignment
                        if (element.assignment != null) {
                            element.assignment.hide()
                        }
                    }
                }
            }
        }

        BBoxAnnotator.prototype.subentries_style_visible = function(parent) {
            var entry, element

            // Make the other entries semi-transparent
            for (var index = 0; index < this.elements.entries.length; index++) {
                entry = this.entries[index]
                element = this.elements.entries[index]
                // If not a sub-entry, skip
                if (entry.parent == null) {
                    continue
                }
                // Check if we are an entry or sub-entry
                if (entry.parent == parent && this.options.subentries.visible != false) {
                    element.bbox.show()
                    // Set the assignment
                    if (element.assignment != null) {
                        if(this.options.subentries.assignments) {
                            element.assignment.show()
                        } else {
                            element.assignment.hide()
                        }
                    }
                } else if (this.options.subentries.visible != true) {
                    element.bbox.hide()
                    // Set the assignment
                    if (element.assignment != null) {
                        element.assignment.hide()
                    }
                }
            }
        }

        BBoxAnnotator.prototype.entries_style_hidden = function(skip) {
            var element

            // Establish the desired parent, if wanted
            skip !== undefined || (skip = null)

            // Make the other entries semi-transparent
            for (var index = 0; index < this.elements.entries.length; index++) {
                if (index != skip) {
                    element = this.elements.entries[index]
                    element.bbox.hide()
                }
            }
        }


        BBoxAnnotator.prototype.entry_style_default = function(index) {
            var entry, element, color_background, color_text

            // Get the appropriate entry element
            entry = this.entries[index]
            element = this.elements.entries[index]

            // Update element CSS
            color_background = entry.highlighted ? this.options.colors.highlighted : this.options.colors.default
            element.bbox.css({
                "border-color": color_background,
                "opacity": "0.8",
                "cursor": "crosshair",
                "background-color": "rgba(0, 0, 0, 0.0)",
                "background-color": "rgba(0, 0, 0, 0.0)",
            })

            // Update the assignment CSS
            if(element.assignment != null) {
                element.assignment.css({
                    "border-color": color_background,
                })
            }

            if(entry.parent == null) {
                element.bbox.css({
                    "outline": "none",
                })
            }
            else {
                element.bbox.css({
                    "outline": this.options.border.width + "px solid " + this.options.colors.subentry,
                })
            }

            color_text = entry.highlighted ? "#FFFFFF" : "#333333"
            element.label.css({
                "background-color": color_background,
                "opacity": "0.8",
                "color": color_text,
            })

            // Update bbox buttons and handles
            element.close.hide()
            element.rotate.hide()
            for (var key in element.resize) {
                element.resize[key].hide()
            }
        }

        BBoxAnnotator.prototype.entry_style_hover = function(index) {
            var entry, element

            // Get the appropriate entry element
            entry = this.entries[index]
            element = this.elements.entries[index]

            // Update element CSS
            element.bbox.css({
                "border-color": this.options.colors.hover,
                "opacity": "1.0",
                "cursor": "move",
                "background-color": "rgba(0, 0, 0, 0.2)",
            })

            // Update the assignment CSS
            if(element.assignment != null) {
                element.assignment.css({
                    "border-color": this.options.colors.hover,
                })
            }

            if(entry.parent == null) {
                element.bbox.css({
                    "outline": "none",
                })
            }
            else {
                element.bbox.css({
                    "outline": this.options.border.width + "px solid " + this.options.colors.subentry,
                })
            }

            element.label.css({
                "background-color": this.options.colors.hover,
                "opacity": "1.0",
                "color": "#333333",
            })

            // Update bbox buttons and handles
            if (this.options.handles.close.enabled) {
                element.close.show()
            }
            if (this.options.handles.rotate.enabled) {
                element.rotate.show()
            }
            for (var key in element.resize) {
                if (this.options.handles.resize.enabled) {
                    element.resize[key].show()
                }
            }
        }

        BBoxAnnotator.prototype.entry_style_focus = function(index) {
            var element

            // Get the appropriate entry element
            entry = this.entries[index]
            element = this.elements.entries[index]

            // Update element CSS
            element.bbox.css({
                "outline": "1000px solid rgba(0, 0, 0, 0.2)",
                "border-color": this.options.colors.focus,
                "opacity": "1.0",
                "cursor": "cell",
                "background-color": "rgba(0, 0, 0, 0.0)",
            })

            // Update the assignment CSS
            if(element.assignment != null) {
                element.assignment.css({
                    "border-color": this.options.colors.focus,
                })
            }

            if(entry.parent != null) {
                element.bbox.css({
                    "outline": this.options.border.width + "px solid " + this.options.colors.subentry,
                })
            }

            element.label.css({
                "background-color": this.options.colors.focus,
                "opacity": "1.0",
                "color": "#FFFFFF",
            })

            // Update bbox buttons and handles
            element.close.hide()
            element.rotate.hide()
            for (var key in element.resize) {
                element.resize[key].hide()
            }
        }

        BBoxAnnotator.prototype.drag_start = function(event) {
            this.state.drag = true
            this.state.which = event.which
            document.onselectstart = function() {
                return false
            }
        }

        BBoxAnnotator.prototype.drag_finish = function() {
            this.state.drag = false
            this.state.which = null
            document.onselectstart = null
        }

        BBoxAnnotator.prototype.register_entry_event_bindings = function(element) {
            var bba

            bba = this

            // Register events for when the box is hovered and unhovered
            element.bbox.hover(function() {
                var index

                index = $(this).prevAll("." + bba.options.classes.bbox).length
                bba.hover_entry(index)
            }, function() {
                bba.hover_entry(null)
            })

            // Register events for when the box is hovered and unhovered
            if(element.assignment != null) {
                element.assignment.hover(function() {
                    var index

                    index = $(this).prevAll("." + bba.options.classes.bbox).length - 1
                    bba.hover_entry(index)
                }, function() {
                    bba.hover_entry(null)
                })
            }

            // Register event for when the close button is selected for a specific bbox
            element.close.mouseup(function(event) {
                var index

                // Find the parent bbox of the close button and the bbox's index
                parent = $(this).parent("." + bba.options.classes.bbox)
                index = parent.prevAll("." + bba.options.classes.bbox).length

                // We want to prevent the selector mode from firing from "free"
                bba.state.mode = "close"

                // Delete the entry from the annotator
                bba.delete_entry_interaction(event, index)
            })

            // Register event for when the close button is selected for a specific bbox
            element.rotate.mousedown(function(event) {
                var parent, index, entry, element

                if (event.which == 3) {
                    return
                }

                // Find the parent bbox of the close button and the bbox's index
                parent = $(this).parent("." + bba.options.classes.bbox)
                index = parent.prevAll("." + bba.options.classes.bbox).length
                entry = bba.entries[index]
                element = bba.elements.entries[index]

                // We want to prevent the selector mode from firing from "free"
                bba.state.mode = "rotate"

                // Set the transparency of other entries
                bba.entries_style_transparent(bba.options.colors.transparent, index)

                // If the
                if (entry.parent != null) {
                    element2 = bba.elements.entries[entry.parent]
                    element2.bbox.css({
                        'opacity': 1.0,
                    })
                }

                // Hide the close and resize buttons
                if ( ! bba.options.debug) {
                    element.label.hide()
                    element.close.hide()
                    for (var key in element.resize) {
                        element.resize[key].hide()
                    }
                }

                document.onselectstart = function() {
                    return false
                }
                bba.anchor_update(event, null)
            })

            for (var key in element.resize) {
                element.resize[key].mousedown(function(event) {
                    if (event.which == 1) {
                        // We want to prevent the selector mode from firing from "free"
                        bba.state.mode = "resize"
                        // Start the resizing
                        bba.resize_start($(this), event)
                    }
                })
            }
        }

        return BBoxAnnotator

    })()

}).call(this)
