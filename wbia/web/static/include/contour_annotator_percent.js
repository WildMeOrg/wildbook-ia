(function() {
    var ContourSelector

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

    ContourSelector = (function() {
        function ContourSelector(frame, existing, options) {
            var options

            options                          !== undefined || (options = {})
            options.prefix                   !== undefined || (options.prefix = "")
            options.ids                      !== undefined || (options.ids = {})
            options.ids.canvas               !== undefined || (options.ids.canvas     = options.prefix + "contour-selector-canvas")
            options.ids.begin                !== undefined || (options.ids.begin      = options.prefix + "contour-selector-begin")
            options.ids.end                  !== undefined || (options.ids.end        = options.prefix + "contour-selector-end")
            options.ids.guiderail            !== undefined || (options.ids.guiderail  = options.prefix + "contour-selector-guiderail")
            options.ids.guideline            !== undefined || (options.ids.guideline  = options.prefix + "contour-selector-guideline")
            options.ids.guidepoint           !== undefined || (options.ids.guidepoint = options.prefix + "contour-selector-guidepoint")
            options.ids.guidearc             !== undefined || (options.ids.guidearc   = options.prefix + "contour-selector-guidearc")
            options.ids.warning              !== undefined || (options.ids.warning    = options.prefix + "contour-selector-warning")
            options.classes                  !== undefined || (options.classes = {})
            options.classes.radius           !== undefined || (options.classes.point = options.prefix + "contour-selector-segment-radius")
            options.classes.segment          !== undefined || (options.classes.point = options.prefix + "contour-selector-segment-point")
            options.colors                   !== undefined || (options.colors = {})
            options.colors.begin             !== undefined || (options.colors.begin       = "#7FFF7F")
            options.colors.radius            !== undefined || (options.colors.radius      = "#2E63FF") // #333333
            options.colors.segment           !== undefined || (options.colors.segment     = "#2E63FF")
            options.colors.highlight         !== undefined || (options.colors.highlight   = "#F0AD4E")
            options.colors.end               !== undefined || (options.colors.end         = "#EB2A18")
            options.colors.guideline         !== undefined || (options.colors.guideline   = "#EB2A18")
            options.colors.guidepoint        !== undefined || (options.colors.guidepoint  = "#EB2A18")
            options.colors.guidetarget       !== undefined || (options.colors.guidetarget = "#333333")
            options.colors.guidearc          !== undefined || (options.colors.guidearc    = "#EB2A18")
            options.colors.guiderail         !== undefined || (options.colors.guiderail   = "#F0AD4E") // #333333
            options.guiderail                !== undefined || (options.guiderail = {})
            options.guiderail.enabled        !== undefined || (options.guiderail.enabled = true)
            options.guiderail.percent        !== undefined || (options.guiderail.percent = 0.0)
            options.size                     !== undefined || (options.size = {})
            options.size.radius              !== undefined || (options.size.radius = 30)
            options.size.segment             !== undefined || (options.size.segment = 4)
            options.size.guiderail           !== undefined || (options.size.guiderail = 1)
            options.size.paint               !== undefined || (options.size.paint = 10)
            options.transparency             !== undefined || (options.transparency = {})
            options.transparency.radius      !== undefined || (options.transparency.radius = 0.3)
            options.transparency.segment     !== undefined || (options.transparency.segment = 1.0)
            options.transparency.guiderail   !== undefined || (options.transparency.guiderail = 0.75)
            options.transparency.guidetarget !== undefined || (options.transparency.guidetarget = 0.25)
            options.warning                  !== undefined || (options.warning = "TOO FAST!<br/>GO BACK TO LAST POINT")
            options.limits                   !== undefined || (options.limits = {})
            options.limits.distance          !== undefined || (options.limits.distance = 20)
            options.limits.arc               !== undefined || (options.limits.arc = {})
            options.limits.arc.degrees       !== undefined || (options.limits.arc.degrees = 45)
            options.limits.arc.radius        !== undefined || (options.limits.arc.radius = 50)
            options.limits.restart           !== undefined || (options.limits.restart = {})
            options.limits.restart.min       !== undefined || (options.limits.restart.min = 3)
            options.limits.restart.max       !== undefined || (options.limits.restart.max = 10)
            options.limits.bounds            !== undefined || (options.limits.bounds = {})
            options.limits.bounds.guiderail  !== undefined || (options.limits.bounds.guiderail = false)
            options.limits.bounds.x          !== undefined || (options.limits.bounds.x = {})
            options.limits.bounds.x.min      !== undefined || (options.limits.bounds.x.min = 0)
            options.limits.bounds.x.max      !== undefined || (options.limits.bounds.x.max = 0)
            options.limits.bounds.y          !== undefined || (options.limits.bounds.y = {})
            options.limits.bounds.y.min      !== undefined || (options.limits.bounds.y.min = 0)
            options.limits.bounds.y.max      !== undefined || (options.limits.bounds.y.max = 0)
            options.debug                    !== undefined || (options.debug = false)
            options.mode                     !== undefined || (options.mode = "edit") // A mode from ["finish", "edit", "draw"]
            /*
                mode == "finish"
                    If the user is going too fast (as dictated by options.limits.distance), then
                    immediately stop drawing and finish the entire contour with an endpoint

                    The user can recover from this mode by moving back to the completed segment
                    and redraw with the normal editing functionality

                mode == "edit" (default)
                    If the user is going too fast, then stop drawing but do not finish the segment.
                    In this incomplete mode, the user can only ESC to cancel the entire segment
                    or complete the existing contour by clicking on the last point

                    The user can recover from this mode by moving back to the last point (in red)
                    and click as if in the normal editing functionality.  The segment is ended
                    with a normal click.

                mode == "draw"
                    If the user is going too fast, then stop drawing but do not finish the segment.
                    In this incomplete mode, the user can only ESC to cancel the entire segment
                    or complete the existing contour by getting within a certain degree of the
                    travel direction and within the distance specified by options.limits.restart.min
                    and options.limits.restart.max.

                    The user can recover from this mode by moving back to the last point (in red)
                    and by placing the cursor within the prescribed arc and distance (shown on
                    screen).  The drawing will automatically resume when the cursor is within this
                    area, no additional clicking is required.  The segment is ended with a normal
                    click.
            */

            // Global attributes
            this.options = options
            this.speeding = false
            this.zoom_modifier = 1.0

            this.points = {
                segment: [],
            }
            this.elements = {
                segment: [],
            }

            this.ctx = null

            // Create objects
            this.create_selector_elements(frame)
            this.create_existing(existing)
        }

        ContourSelector.prototype.resize = function(frame) {
            var width, height, point

            width = this.elements.frame.width()
            height = this.elements.frame.height()

            // This clears the canvas, redraw canvas
            this.elements.canvas.attr("width", width)
            this.elements.canvas.attr("height", height)

            canvas_style = {
                "width": width + "px",
                "height": height + "px",
            }
            this.elements.canvas.css(canvas_style)

            // Create the context
            this.ctx = this.elements.canvas[0].getContext('2d')

            // Clear the context
            this.ctx.clearRect(0, 0, this.elements.canvas.width(), this.elements.canvas.height());
            for (var index = 0; index < this.points.segment.length; index++) {
                point = this.points.segment[index]
                this.draw_segment(point, true)
            }
        }

        ContourSelector.prototype.create_selector_elements = function(frame) {
            var width, height, padding
            var endpoint_style, guideline_style, guidepoint_style, warning_style

            // Define the image frame as an element
            this.elements.frame = frame

            width = this.elements.frame.width()
            height = this.elements.frame.height()

            canvas_style = {
                "opacity": this.options.transparency.radius
            }
            // Create the draw canvas
            this.elements.canvas = $('<canvas id="' + this.options.ids.canvas + '"></canvas>')
            this.elements.canvas.css(canvas_style)
            this.elements.frame.append(this.elements.canvas)
            this.resize()

            // Begin point
            endpoint_style = {
                "top": "0",
                "left": "0",
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
            this.elements.end = $('<div id="' + this.options.ids.end + '"></div>')
            this.elements.end.css(endpoint_style)
            this.elements.end.css({
                "background-color": this.options.colors.end,
            })
            this.elements.frame.append(this.elements.end)
            this.elements.end.hide()

            // Begin guiderail
            this.options.padding = this.options.guiderail.percent / (1.0 + (2.0 * this.options.guiderail.percent))
            guideline_style = {
                "position": "absolute",
                "top":    (100.0 * this.options.padding) + "%",
                "bottom": (100.0 * this.options.padding) + "%",
                "left":   (100.0 * this.options.padding) + "%",
                "right":  (100.0 * this.options.padding) + "%",
                "border": this.options.size.guiderail + "px solid " + this.options.colors.guiderail,
                "color": this.options.colors.guiderail,
                "opacity": this.options.transparency.guiderail
            }

            this.elements.guiderail = $('<div id="' + this.options.ids.guiderail + '"></div>')
            this.elements.guiderail.css(guideline_style)
            this.elements.frame.append(this.elements.guiderail)

            // Begin guideline
            guideline_style = {
                "border": "1px solid " + this.options.colors.guideline,
                "position": "absolute",
                "transform-origin": "0% 0%",
                "color": this.options.colors.guideline,
            }

            this.elements.guideline = $('<div id="' + this.options.ids.guideline + '"></div>')
            this.elements.guideline.css(guideline_style)
            this.elements.frame.append(this.elements.guideline)
            this.elements.guideline.hide()

            guidepoint_style = {
                "top": "0",
                "left": "0",
                "width": "6px",
                "height": "6px",
                "position": "absolute",
                "border-radius": "6px",
                "border": "1px solid #fff",
                "margin-top": "-4px",
                "margin-left": "-4px",
                "z-index": "2",
            }

            // Begin guidepoint
            this.elements.guidepoint = $('<div id="' + this.options.ids.guidepoint + '"></div>')
            this.elements.guidepoint.css(guidepoint_style)
            this.elements.guidepoint.css({
                "background-color": this.options.colors.guidepoint,
            })

            radius = this.options.size.paint
            radius_double = Math.floor(radius * 2.0)
            guidetarget_style = {
                "top": "0",
                "left": "0",
                "width": radius_double + "px",
                "height": radius_double + "px",
                "position": "absolute",
                "border-radius": radius_double + "px",
                "border": "1px solid " + this.options.colors.guidetarget,
                "margin-top": "-" + (radius - 2) + "px",
                "margin-left": "-" + (radius - 2) + "px",
                "z-index": "10",
                "opacity": this.options.transparency.guidetarget,
            }

            // Begin guidepoint
            this.elements.guidetarget = $('<div id="' + this.options.ids.guidepoint + '"></div>')
            this.elements.guidetarget.css(guidetarget_style)
            this.elements.guidepoint.append(this.elements.guidetarget)
            this.elements.frame.append(this.elements.guidepoint)
            this.elements.guidepoint.hide()

            radius = Math.floor(this.options.limits.arc.radius)
            radius_double = Math.floor(radius * 2.0)
            guidearc_style = {
                "position": "absolute",
                "top": "0",
                "left": "0",
                "margin-left": "-" + radius + "px",
                "margin-top": "-" + radius + "px",
                "opacity": "0.3",
                "transform-origin": "50% 50%",
            }

            // Begin guidearc
            this.elements.guidearc = $('<div id="' + this.options.ids.guidearc + '"></div>')
            this.elements.guidearc.css(guidearc_style)

            degrees = Math.floor(90 - (this.options.limits.arc.degrees / 2.0))
            guidearc_arc_style = {
                "position": "relative",
                "font-size": "120px",
                "width": radius_double + "px",
                "height": radius_double + "px",
                "-webkit-border-radius": "50%",
                "-moz-border-radius": "50%",
                "-ms-border-radius": "50%",
                "-o-border-radius": "50%",
                "border-radius": "50%",
                "float": "left",
                "background-color": "transparent",
                "-webkit-transform": "rotate(" + degrees + "deg)",
                "-moz-transform": "rotate(" + degrees + "deg)",
                "-ms-transform": "rotate(" + degrees + "deg)",
                "-o-transform": "rotate(" + degrees + "deg)",
                "transform": "rotate(" + degrees + "deg)",
            }

            // Begin guidearc
            this.elements.guidearc_arc = $('<div id="' + this.options.ids.guidearc + '-arc"></div>')
            this.elements.guidearc_arc.css(guidearc_arc_style)
            this.elements.guidearc.append(this.elements.guidearc_arc)

            guidearc_slice_style = {
                "position": "absolute",
                "width": radius_double + "px",
                "height": radius_double + "px",
                "clip": "rect(0px, " + radius_double + "px, " + radius_double + "px, " + radius + "px)",
            }

            // Begin guidearc slice
            this.elements.guidearc_slice = $('<div id="' + this.options.ids.guidearc + '-slice"></div>')
            this.elements.guidearc_slice.css(guidearc_slice_style)
            this.elements.guidearc_arc.append(this.elements.guidearc_slice)

            degrees = Math.floor(this.options.limits.arc.degrees)
            guidearc_bar_style = {
                "position": "absolute",
                "border": radius + "px solid red",
                "width": "0px",
                "height": "0px",
                "clip": "rect(0px, " + radius + "px, " + radius_double + "px, 0px)",
                "-webkit-border-radius": "50%",
                "-moz-border-radius": "50%",
                "-ms-border-radius": "50%",
                "-o-border-radius": "50%",
                "border-radius": "50%",
                "-webkit-transform": "rotate(" + degrees + "deg)",
                "-moz-transform": "rotate(" + degrees + "deg)",
                "-ms-transform": "rotate(" + degrees + "deg)",
                "-o-transform": "rotate(" + degrees + "deg)",
                "transform": "rotate(" + degrees + "deg)",
            }
            this.elements.guidearc_bar = $('<div id="' + this.options.ids.guidearc + '-bar"></div>')
            this.elements.guidearc_bar.css(guidearc_bar_style)
            this.elements.guidearc_slice.append(this.elements.guidearc_bar)
            this.elements.frame.append(this.elements.guidearc)
            this.elements.guidearc.hide()

            // Begin warning
            warning_style = {
                "position": "absolute",
                "top": "0",
                "bottom": "0",
                "left": "0",
                "right": "0",
                "border": "5px solid " + this.options.colors.guidepoint,
            }

            this.elements.warning = $('<div id="' + this.options.ids.warning + '"></div>')
            this.elements.warning.css(warning_style)

            message_style = {
                "position": "relative",
                "width": "100%",
                "color": "#FFFFFF",
                "text-align": "center",
                "background-color": this.options.colors.guidepoint,
            }

            this.elements.message = $('<div id="' + this.options.ids.warning + '-message">            <b>' + this.options.warning + '</b></div>')
            this.elements.message.css(message_style)
            this.elements.warning.append(this.elements.message)
            this.elements.frame.append(this.elements.warning)
            this.elements.warning.hide()
        }

        ContourSelector.prototype.check_boundaries = function(event, point) {
            var offset, point, bounds, width, height

            point !== undefined || (point = null)

            width = this.elements.frame.width()
            height = this.elements.frame.height()

            if (event == null) {
                return null
            }

            if (point == null) {
                offset = this.elements.frame.offset()
                point = {
                    x: event.pageX - offset.left,
                    y: event.pageY - offset.top,
                }
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

            if (this.options.limits.bounds.guiderail) {
                bounds.x.min -= this.options.padding * width
                bounds.x.max -= this.options.padding * width
                bounds.y.min -= this.options.padding * height
                bounds.y.max -= this.options.padding * height
            }

            // Check lower boundaries
            point.x = Math.max(point.x, 0 - bounds.x.min)
            point.y = Math.max(point.y, 0 - bounds.y.min)
            // Check upper boundaries
            point.x = Math.min(point.x, width - 1 + bounds.x.max)
            point.y = Math.min(point.y, height - 1 + bounds.y.max)
            return point
        }

        ContourSelector.prototype.create_existing = function(existing) {
            invalid = false
            invalid = invalid || (existing.segment === undefined)
            invalid = invalid || (existing.segment.length === undefined)
            invalid = invalid || (existing.begin === undefined)
            invalid = invalid || (existing.end === undefined)

            console.log(existing)
            if ( ! invalid) {

                invalid = invalid || (existing.start <= 0)
                invalid = invalid || (existing.end <= 0)
                invalid = invalid || (existing.segment.length <= existing.start)
                invalid = invalid || (existing.segment.length <= existing.end)


                for (var index = 0; index < existing.segment.length; index++) {
                    point = existing.segment[index]
                    invalid = invalid || (point.x    === undefined)
                    invalid = invalid || (point.y    === undefined)
                    invalid = invalid || (point.r    === undefined)
                    invalid = invalid || (point.flag === undefined)

                    if (invalid) {
                        break
                    }
                }
            }

            if (invalid) {
                existing = null
            }

            // Index into segment for begin and end
            if (existing != null) {

                // Convert compact format into expanded format
                width = this.elements.frame.width()
                height = this.elements.frame.height()

                padding_w = this.options.padding * width
                padding_h = this.options.padding * height

                local_width  = width -  2.0 * padding_w
                local_height = height - 2.0 * padding_h

                for (var index = 0; index < existing.segment.length; index++) {
                    existing_point = existing.segment[index]

                    point = {
                        "x": {
                            "local":  existing_point.x,
                        },
                        "y": {
                            "local":  existing_point.y,
                        },
                        "radius": {
                            "local":  existing_point.r,
                        },
                        "highlight":  existing_point.flag,
                    }

                    point.x.pixel      = (point.x.local * local_width) + padding_w
                    point.x.global     = point.x.pixel / width

                    point.y.pixel      = (point.y.local * local_height) + padding_h
                    point.y.global     = point.y.pixel / height

                    point.radius.pixel  = point.radius.local * local_width
                    point.radius.global = point.radius.pixel / width

                    existing.segment[index] = point
                }

                existing.begin = existing.segment[existing.begin]
                existing.end   = existing.segment[existing.end]

                console.log(existing.begin)
                console.log(existing.end)
            }

            this.existing = existing
            this.draw_existing()
        }

        ContourSelector.prototype.draw_existing = function() {
            this.clear()

            if (this.existing == null) {
                return
            }

            this.points.segment = this.existing.segment
            this.points.begin   = this.existing.begin
            this.points.end     = this.existing.end

            this.elements.begin.show()
            this.elements.begin.css({
                "top":  (100.0 * this.points.begin.y.global) + "%",
                "left": (100.0 * this.points.begin.x.global) + "%",
            })

            this.elements.end.show()
            this.elements.end.css({
                "top":  (100.0 * this.points.end.y.global) + "%",
                "left": (100.0 * this.points.end.x.global) + "%",
            })

            for (var index = 0; index < this.points.segment.length; index++) {
                point = this.points.segment[index]
                this.draw_segment(point)
            }
        }

        ContourSelector.prototype.hide_warning = function() {
            this.elements.warning.hide()
        }

        ContourSelector.prototype.clear = function(keep_index) {

            keep_index !== undefined || (keep_index = null)

            if (keep_index < 0 || this.points.segment.length <= keep_index) {
                keep_index = null
            }

            this.speeding = false
            this.elements.end.hide()
            this.elements.begin.hide()
            this.elements.guideline.hide()
            this.elements.guidepoint.hide()
            this.elements.guidearc.hide()

            this.ctx.clearRect(0, 0, this.elements.canvas.width(), this.elements.canvas.height());

            for (var index = 0; index < this.elements.segment.length; index++) {
                if (keep_index == null || keep_index <= index) {
                    this.elements.segment[index].detach()
                } else {
                    point = this.points.segment[index]
                    this.draw_segment(point, true)
                }
            }

            if (keep_index == null) {
                this.points.segment = []
                this.elements.segment = []
            } else {
                this.points.segment.splice(keep_index + 1)
                this.elements.segment.splice(keep_index + 1)
            }
        }

        ContourSelector.prototype.current_distance = function(index) {
            var width, height, point, diff_x, diff_y, dist

            index !== undefined || (index = this.points.segment.length - 1)

            if (index < 0) {
                return [-1, null, null]
            }

            width = this.elements.frame.width()
            height = this.elements.frame.height()

            point = this.points.segment[index]
            diff_x = this.points.cursor.x - (point.x.global * width)
            diff_y = this.points.cursor.y - (point.y.global * height)
            dist = Math.sqrt(Math.pow(diff_x, 2) + Math.pow(diff_y, 2))

            return [dist, point]
        }

        ContourSelector.prototype.start = function(event, index) {
            var width, height, data

            index !== undefined || (index = null)

            width = this.elements.frame.width()
            height = this.elements.frame.height()

            this.points.cursor = this.check_boundaries(event)

            if (index != null) {
                values = this.current_distance(index)
                distance = values[0]
                if (distance > this.options.limits.restart.min) {
                    return false
                }
            } else {
                data = this.get_selector_data()
                if (data != null) {
                    return false
                }
            }

            this.clear(index)

            // Add starting point to the segment
            point = this.add_segment(event)

            if (point == null) {
                if (this.points.segment.length > 0) {
                    point = this.points.segment[this.points.segment.length - 1]
                } else {
                    console.log('ASSERTION ERROR! STARTING POINT CANNOT BE NULL')
                }
            }

            if (index == null) {
                this.points.begin = point
            }

            // Show selectors, as needed
            this.elements.begin.show()
            this.elements.begin.css({
                "top":  (100.0 * this.points.begin.y.global) + "%",
                "left": (100.0 * this.points.begin.x.global) + "%",
            })

            this.elements.frame.css({
                "cursor": "crosshair"
            })

            document.onselectstart = function() {
                return false
            }

            return true
        }

        ContourSelector.prototype.declutter = function(flag) {
            console.log('DECLUTTER: ' + flag)
            if (flag) {
                css_style = {
                    'opacity': 0.0,
                }

                radius      = 0.0
                segment     = 0.0
            } else {
                css_style = {
                    'opacity': 1.0,
                }

                radius      = this.options.transparency.radius
                segment     = this.options.transparency.segment
            }

            this.elements.end.css(css_style)
            this.elements.begin.css(css_style)

            cta.cts.elements.canvas.css({
                'opacity': radius,
            })

            for (var index = 0; index < this.elements.segment.length; index++) {
                this.elements.segment[index].css({
                    'opacity': segment,
                })
            }
        }

        ContourSelector.prototype.add_segment = function(event) {
            var width, height, padding_w, padding_h, highlight, point

            width = this.elements.frame.width()
            height = this.elements.frame.height()

            // This functuion requires that points.cursor has already been updated
            padding_w = this.options.padding * width
            padding_h = this.options.padding * height

            local_width  = width -  2.0 * padding_w
            local_height = height - 2.0 * padding_h

            highlight = event.shiftKey
            point = {
                "x": {
                    "pixel":  this.points.cursor.x,
                    "global": this.points.cursor.x / width,
                },
                "y": {
                    "pixel":  this.points.cursor.y,
                    "global": this.points.cursor.y / height,
                },
                "radius": {
                    "pixel":   this.options.size.radius,
                    "global":  this.options.size.radius / width,
                },
                "highlight": highlight,
            }

            point.x.local      = (point.x.pixel - padding_w) / local_width
            point.y.local      = (point.y.pixel - padding_h) / local_height
            point.radius.local = point.radius.pixel          / local_width

            if (this.points.segment.length > 0) {
                last_index = this.points.segment.length - 1
                last_point = this.points.segment[last_index]

                if (last_point.x.local == point.x.local) {
                    if (last_point.y.local == point.y.local) {
                        if (last_point.radius.local == point.radius.local) {
                            if (last_point.highlight == point.highlight) {
                                return null
                            }
                        }
                    }
                }
            }

            this.points.segment.push(point)
            return point
        }

        ContourSelector.prototype.draw_segment = function(point, radius_only) {
            var width, height, temp

            radius_only !== undefined || (radius_only = false)

            if (point == null) {
                return
            }

            width = this.elements.frame.width()
            height = this.elements.frame.height()

            temp = {
                "x": point.x.global * width,
                "y": point.y.global * height,
                "radius": point.radius.global * width,
            }

            // Draw radius on canvas
            this.ctx.beginPath()
            this.ctx.fillStyle = this.options.colors.radius
            this.ctx.ellipse(temp.x, temp.y, temp.radius, temp.radius, 0, 0, 2 * Math.PI)
            this.ctx.fill()

            if (radius_only) {
                return
            }

            size = this.options.size.segment
            size_half = size / 2.0
            segment_style = {
                "top": (100.0 * point.y.global) + "%",
                "left": (100.0 * point.x.global) + "%",
                "width": size + "px",
                "height": size + "px",
                "position": "absolute",
                "border-radius": size + "px",
                "margin-top": "-" + size_half + "px",
                "margin-left": "-" + size_half + "px",
                "opacity": this.options.transparency.segment,
                "z-index": "1",
            }

            // Begin segment center point
            segment = $('<div class="' + this.options.classes.segment + '"></div>')
            segment.css(segment_style)
            this.elements.frame.append(segment)
            this.elements.segment.push(segment)

            index = this.elements.segment.length - 1
            this.color_segment(index)
        }

        ContourSelector.prototype.color_segment = function(index) {
            var point, color, segment

            point = this.points.segment[index]

            if (point.highlight) {
                color = this.options.colors.highlight
            } else {
                color = this.options.colors.segment
            }

            segment = this.elements.segment[index]
            segment.css({
                "background-color": color,
            })

        }
        ContourSelector.prototype.update_cursor = function(event, skip) {
            var width, height, point
            var opacity, css_theta, css_rotate
            var highlight, point, size, size_half
            var segment, segment_style, finish

            skip !== undefined || (skip = false)

            // Update the selector based on the current cursor location
            this.points.cursor = this.check_boundaries(event)

            width = this.elements.frame.width()
            height = this.elements.frame.height()

            values = this.current_distance()
            distance = values[0]
            last = values[1]

            point = {
                "x" : (last.x.global * width),
                "y" : (last.y.global * height),
            }
            css_theta = calculate_angle(point, this.points.cursor)
            css_theta_ma = null

            if (this.speeding && this.options.mode == "draw") {
                // Calculate the Moving Average angle between the last X points
                rate = 0.8
                samples = 5
                previous = null
                for (var offset = samples; offset >= 0; offset--) {
                    index = this.points.segment.length - 1 - offset
                    if (0 <= index && index < this.points.segment.length) {
                        point = this.points.segment[index]
                        point = {
                            "x" : (point.x.global * width),
                            "y" : (point.y.global * height),
                        }

                        if (previous == null) {
                            previous = point
                        } else if (point != null) {
                            angle = calculate_angle(previous, point)
                            if (css_theta_ma == null) {
                                css_theta_ma = angle
                            } else {
                                css_theta_ma = angle * rate + css_theta_ma * (1.0 - rate)
                            }
                            previous = point
                        }
                    }
                }
            }

            max_distance = this.options.limits.distance
            max_distance *= this.zoom_modifier

            if (! this.speeding && distance > this.options.limits.distance) {
                this.speeding = true
            } else if (this.speeding) {
                if (this.options.mode == "draw") {
                    if(this.options.limits.restart.min <= distance && distance <= this.options.limits.restart.max) {
                        margin = Math.PI * 2.0 * (22.0 / 360.0)
                        diff = css_theta_ma - css_theta

                        css_theta_min = Math.min(css_theta_ma, css_theta)
                        css_theta_max = Math.max(css_theta_ma, css_theta)
                        diff = css_theta_max - css_theta_min

                        if (diff > Math.PI) {
                            diff -= Math.PI
                            diff = Math.PI - diff
                        }

                        if(diff <= margin) {
                            this.speeding = false
                        }
                    }
                } else if (this.options.mode == "edit") {

                }
            }

            finish = false
            if (this.speeding && this.options.mode == "finish") {
                this.speeding = false
                finish = true
                this.elements.warning.show()

                // We need to update the cursor to use the last good segment location, not the current cursor
                index = this.points.segment.length - 1
                point = this.points.segment[index]
                point = {
                    "x": point.x.global * width,
                    "y": point.y.global * height,
                }
                this.points.cursor = this.check_boundaries(event, point)
            }

            if (this.speeding) {

                if (distance <= this.options.limits.restart.min) {
                    cursor_style = "cell"
                } else {
                    cursor_style = "crosshair"
                }

                this.elements.frame.css({
                    "cursor": cursor_style
                })

                opacity = Math.max(0.50, Math.min(1.0, distance / 200.0))
                this.elements.guideline.css({
                    "top": (100.0 * last.y.global) + "%",
                    "left": (100.0 * last.x.global) + "%",
                    "width": distance + "px",
                    "opacity": opacity,
                })

                this.elements.guidepoint.css({
                    "top": (100.0 * last.y.global) + "%",
                    "left": (100.0 * last.x.global) + "%",
                    "opacity": 1.0,
                })

                this.elements.guidearc.css({
                    "top": (100.0 * last.y.global) + "%",
                    "left": (100.0 * last.x.global) + "%",
                    "opacity": opacity,
                })

                css_rotate = {
                    "transform": "rotate(" + css_theta + "rad)",
                    "msTransform": "rotate(" + css_theta + "rad)",
                    "-o-transform": "rotate(" + css_theta + "rad)",
                    "-ms-transform": "rotate(" + css_theta + "rad)",
                    "-moz-transform": "rotate(" + css_theta + "rad)",
                    "-sand-transform": "rotate(" + css_theta + "rad)",
                    "-webkit-transform": "rotate(" + css_theta + "rad)",
                }

                this.elements.guideline.css(css_rotate)

                if (css_theta_ma == null) {
                    this.elements.guidearc.hide()
                } else {
                    css_rotate = {
                        "transform": "rotate(" + css_theta_ma + "rad)",
                        "msTransform": "rotate(" + css_theta_ma + "rad)",
                        "-o-transform": "rotate(" + css_theta_ma + "rad)",
                        "-ms-transform": "rotate(" + css_theta_ma + "rad)",
                        "-moz-transform": "rotate(" + css_theta_ma + "rad)",
                        "-sand-transform": "rotate(" + css_theta_ma + "rad)",
                        "-webkit-transform": "rotate(" + css_theta_ma + "rad)",
                    }
                    this.elements.guidearc.css(css_rotate)
                    this.elements.guidearc.show()
                }

                this.elements.guideline.show()
                this.elements.guidepoint.show()
                this.elements.guidetarget.hide()
                this.elements.warning.show()
            } else {
                // Not speeding
                this.elements.guideline.hide()
                this.elements.guidepoint.hide()
                this.elements.guidetarget.hide()
                this.elements.guidearc.hide()
                if ( ! finish) {
                    this.elements.warning.hide()
                }

                this.elements.frame.css({
                    "cursor": "crosshair"
                })

                // Add to the current segment and draw it
                if ( ! skip) {
                    point = this.add_segment(event)
                    this.draw_segment(point)
                }
            }

            return finish
        }

        ContourSelector.prototype.update_anchor = function(event) {
            var width, height, point, closest_distance, closest_index, reference
            var opacity, css_theta, css_rotate
            var highlight, point, size, size_half
            var segment, segment_style

            // Update the selector based on the current cursor location
            this.points.cursor = this.check_boundaries(event)

            width = this.elements.frame.width()
            height = this.elements.frame.height()

            closest_distance = Infinity
            closest_index = null
            for (var index = 0; index < this.points.segment.length; index++) {
                values = this.current_distance(index)
                distance = values[0]
                if (distance < closest_distance && distance < this.options.size.radius) {
                    closest_distance = distance
                    closest_index = index
                }
            }

            if (closest_index !== null) {

                if (event.shiftKey || event.ctrlKey) {
                    highlight = event.shiftKey

                    margin = 50
                    reference = this.points.segment[closest_index]
                    for (var index = closest_index - margin; index <= closest_index + margin; index++) {
                        if (0 <= index && index < this.points.segment.length) {
                            point = this.points.segment[index]
                            diff_x = (reference.x.global - point.x.global) * width
                            diff_y = (reference.y.global - point.y.global) * height
                            dist = Math.sqrt(Math.pow(diff_x, 2) + Math.pow(diff_y, 2))
                            if (dist <= this.options.size.paint) {
                                this.points.segment[index].highlight = highlight
                                this.color_segment(index)
                            }
                        }
                    }
                }

                closest_values = this.current_distance(closest_index)
                distance = closest_values[0]
                last = closest_values[1]

                if (distance <= this.options.limits.restart.min) {
                    cursor_style = "cell"
                } else {
                    cursor_style = "crosshair"
                }

                this.elements.frame.css({
                    "cursor": cursor_style
                })

                this.elements.guidepoint.css({
                    "top": (100.0 * last.y.global) + "%",
                    "left": (100.0 * last.x.global) + "%",
                    "opacity": 1.0,
                })
                this.elements.guidepoint.show()
                this.elements.guidetarget.show()
            } else {
                this.elements.guidepoint.hide()
                this.elements.guidetarget.show()
            }

            return closest_index
        }

        ContourSelector.prototype.update_radius = function(radius) {
            this.options.size.radius = radius
            return this.options.size.radius
        }

        ContourSelector.prototype.finish = function(event, update) {
            var data, width, height
            if (event == null) {
                this.clear()
                return true
            }

            if (update) {
                this.points.cursor = this.check_boundaries(event)
            }

            if (this.speeding) {
                values = this.current_distance()
                distance = values[0]

                if(this.speeding && this.options.mode == "edit") {
                    if (distance <= this.options.limits.restart.min) {
                        this.speeding = false
                        this.update_cursor(event, true)
                    }
                    return false
                }

                this.clear()
                return true
            }

            width = this.elements.frame.width()
            height = this.elements.frame.height()

            // Add ending point to the segment
            point = this.add_segment(event)
            if (point == null) {
                point = this.points.segment[this.points.segment.length - 1]
            }
            this.points.end = point

            this.elements.end.show()
            this.elements.end.css({
                "top":  (100.0 * this.points.end.y.global) + "%",
                "left": (100.0 * this.points.end.x.global) + "%",
            })

            $("body").css({
                "cursor": "default"
            })

            // Reset the global cursor mode and re-allow highlight selection
            document.onselectstart = null
            return true
        }

        ContourSelector.prototype.get_selector_data = function() {
            var data

            if (this.points.segment.length == 0) {
                data = null
            } else {
                data = {
                    begin:   0,
                    end:     this.points.segment.length -1,
                    segment: [],
                }

                for (var index = 0; index < this.points.segment.length; index++) {
                    point = this.points.segment[index]
                    point_data = {
                        "x": point.x.local,
                        "y": point.y.local,
                        "r": point.radius.local,
                        "flag": point.highlight,
                    }
                    data.segment.push(point_data)
                }
            }

            return data
        }

        ContourSelector.prototype.refresh = function(show_guiderail) {
            show_guiderail !== undefined || (show_guiderail = this.options.guiderail.enabled)
            this.options.guiderail.enabled = show_guiderail
            // this.options.limits.bounds.guiderail = this.options.guiderail.enabled

            if (this.options.guiderail.enabled) {
                this.elements.guiderail.show()
            } else {
                this.elements.guiderail.hide()
            }

            data = this.get_selector_data()
            return data
        }

        return ContourSelector
    })()

    this.ContourAnnotator = (function() {
        function ContourAnnotator(url, existing, options) {
            var options

            options                              !== undefined || (options = {})
            options.prefix                       !== undefined || (options.prefix = "")
            options.ids                          !== undefined || (options.ids = {})
            options.ids.container                !== undefined || (options.ids.container = options.prefix + "contour-annotator-container")
            options.props                        !== undefined || (options.props = {})
            options.colors                       !== undefined || (options.colors = {})
            options.colors.default               !== undefined || (options.colors.default = "#7FFF7F")
            options.actions                      !== undefined || (options.actions = {})
            options.actions.click                !== undefined || (options.actions.click = {})
            options.actions.click.right          !== undefined || (options.actions.click.right = true)
            options.hotkeys                      !== undefined || (options.hotkeys = {})
            options.hotkeys.enabled              !== undefined || (options.hotkeys.enabled = true)
            options.hotkeys.delete               !== undefined || (options.hotkeys.delete    = [8, 46, 75])
            options.hotkeys.exit                 !== undefined || (options.hotkeys.exit      = [27])
            options.hotkeys.zoom                 !== undefined || (options.hotkeys.zoom      = [90])
            options.hotkeys.declutter            !== undefined || (options.hotkeys.declutter = [72])
            options.zoom                         !== undefined || (options.zoom = {})
            options.zoom.enabled                 !== undefined || (options.zoom.enabled = true)
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
            options.confirm.reset                !== undefined || (options.confirm.reset = true)
            options.callbacks                    !== undefined || (options.callbacks = {})
            options.callbacks.onload             !== undefined || (options.callbacks.onload = null)
            options.callbacks.onselector         !== undefined || (options.callbacks.onselector = null)
            options.callbacks.onchange           !== undefined || (options.callbacks.onchange = null)
            options.callbacks.ondelete           !== undefined || (options.callbacks.ondelete = null)
            options.debug                        !== undefined || (options.debug = false)

            // Global attributes
            this.existing = existing
            this.options = options

            this.elements = []

            // Keep track of the current state of the annotator
            this.state = {
                mode:     "free",
                inside:   false,
                hover:    null,
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
                cta.cts = new ContourSelector(cta.elements.frame, cta.existing, cta.options)

                cta.refresh()

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

        ContourAnnotator.prototype.delete_interaction = function(event) {
            if (this.options.confirm.delete) {
                response = confirm("Are you sure you want to delete this segment?")
                if ( ! response) {
                    return
                }
            }

            cta.delete_current()
        }

        ContourAnnotator.prototype.reset_interaction = function(event) {
            if (this.options.confirm.reset) {
                response = confirm("Are you sure you want to reset this segment back to original?")
                if ( ! response) {
                    return
                }
            }

            cta.reset_existing(event)
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

                if (cta.options.hotkeys.declutter.indexOf(key) != -1) {
                    cta.cts.declutter(true)
                }

                if (cta.options.hotkeys.enabled) {

                    // Delete key pressed
                    if (cta.options.hotkeys.delete.indexOf(key) != -1) {
                        if (cta.state.mode == "selector") {
                            cta.selector_cancel(event)
                        } else {
                            cta.delete_interaction(event)
                        }
                    }

                    // Escape key pressed
                    if (cta.options.hotkeys.exit.indexOf(key) != -1) {
                        if (cta.state.mode == "selector") {
                            cta.selector_cancel(event)
                        }
                        cta.cts.hide_warning()
                    }
                }
            })

            $(window).keyup(function(event) {
                var key, hotkeys_movement

                key = event.which

                if (cta.options.hotkeys.declutter.indexOf(key) != -1) {
                    cta.cts.declutter(false)
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
                } else if (cta.state.mode == "free") {
                    cta.selector_anchor(event)
                }
            })

            // Catch the mouse up event (wherever it may appear on the screen)
            $(window).mouseup(function(event) {
                if (event.which == 1 || event.which == 3) {
                    if (cta.state.mode == "free" && cta.state.inside) {
                        if (event.which == 3 && cta.options.actions.click.right) {
                            cta.delete_current()
                        }
                        cta.selector_start(event)
                        cta.cts.hide_warning()
                    } else if (cta.state.mode == "selector") {
                        cta.selector_finish(event)
                    }
                }

                cta.state.inside = false
            })
        }

        ContourAnnotator.prototype.update_radius = function(radius) {
            if (this.cts === undefined) {
                radius = null
            } else {
                radius = this.cts.update_radius(radius)
            }
            return radius
        }

        ContourAnnotator.prototype.resize = function() {
            var width, height, w1, h1, w2, h2, offset, scroll, left, margin

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

            // Update console
            if(this.cts !== undefined) {
                this.cts.resize()
            }

            return this.refresh()
        }

        ContourAnnotator.prototype.zoom_start = function(event) {
            this.state.zoom = this.options.zoom.enabled

            w1 = this.elements.image.width
            w2 = this.elements.frame.width()
            this.cts.zoom_modifier = w1 / w2

            this.selector_cancel(null)
            this.resize()
        }

        ContourAnnotator.prototype.zoom_finish = function(event) {
            this.state.zoom = false
            this.cts.zoom_modifier = 1.0
            this.selector_cancel(null)
            this.resize()
        }

        ContourAnnotator.prototype.refresh = function(show_guiderail) {
            if (this.cts === undefined) {
                data = null
            } else {
                data = this.cts.refresh(show_guiderail)
            }

            if (this.options.callbacks.onchange != null) {
                this.options.callbacks.onchange(data)
            }

            return data
        }

        ContourAnnotator.prototype.delete_current = function() {
            this.cts.clear()

            // Refresh the display
            this.refresh()

            // Trigger ondelete
            if (cta.options.callbacks.ondelete != null) {
                return cta.options.callbacks.ondelete()
            }
        }

        ContourAnnotator.prototype.selector_start = function(event) {
            // Start the ContourSelector
            started = this.cts.start(event, this.state.hover)

            if (started) {
                this.state.mode = "selector"

                // notify on adding segment from selection
                if (this.options.callbacks.onselector != null) {
                    return this.options.callbacks.onselector()
                }
            }
        }

        ContourAnnotator.prototype.selector_update = function(event) {
            var finish

            // Update the ContourSelector
            finish = this.cts.update_cursor(event)

            if (finish) {
                this.selector_finish(event, false)
            }
        }

        ContourAnnotator.prototype.selector_anchor = function(event) {
            // Update the ContourSelector
            this.state.hover = this.cts.update_anchor(event)
        }

        ContourAnnotator.prototype.selector_cancel = function(event) {
            this.selector_finish(null)
        }

        ContourAnnotator.prototype.selector_finish = function(event, update) {

            update !== undefined || (update = true)

            // If we are in the "selector" mode, finish the ContourSelector and add the segment
            if (this.state.mode == "selector") {
                // Get the segment data by finishing the ContourSelector
                complete = this.cts.finish(event, update)

                if (complete) {
                    // Release the mode to "free"
                    this.state.mode = "free"
                    // Update the data
                    this.refresh()
                }
            }
        }

        ContourAnnotator.prototype.reset_existing = function(event) {
            this.cts.draw_existing()
            this.refresh()
        }

        return ContourAnnotator

    })()

}).call(this)
