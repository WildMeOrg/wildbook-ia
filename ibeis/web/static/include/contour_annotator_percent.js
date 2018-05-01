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

            options                       !== undefined || (options = {})
            options.prefix                !== undefined || (options.prefix = "")
            options.ids                   !== undefined || (options.ids = {})
            options.ids.canvas            !== undefined || (options.ids.canvas     = options.prefix + "contour-selector-canvas")
            options.ids.begin             !== undefined || (options.ids.begin      = options.prefix + "contour-selector-begin")
            options.ids.end               !== undefined || (options.ids.end        = options.prefix + "contour-selector-end")
            options.ids.guiderail         !== undefined || (options.ids.guiderail  = options.prefix + "contour-selector-guiderail")
            options.ids.guideline         !== undefined || (options.ids.guideline  = options.prefix + "contour-selector-guideline")
            options.ids.guidepoint        !== undefined || (options.ids.guidepoint = options.prefix + "contour-selector-guidepoint")
            options.ids.warning           !== undefined || (options.ids.warning    = options.prefix + "contour-selector-warning")
            options.classes               !== undefined || (options.classes = {})
            options.classes.radius        !== undefined || (options.classes.point = options.prefix + "contour-selector-segment-radius")
            options.classes.segment       !== undefined || (options.classes.point = options.prefix + "contour-selector-segment-point")
            options.colors                !== undefined || (options.colors = {})
            options.colors.begin          !== undefined || (options.colors.begin      = "#7FFF7F")
            options.colors.radius         !== undefined || (options.colors.radius     = "#333333")
            options.colors.segment        !== undefined || (options.colors.segment    = "#2E63FF")
            options.colors.highlight      !== undefined || (options.colors.highlight  = "#F0AD4E")
            options.colors.end            !== undefined || (options.colors.end        = "#EB2A18")
            options.colors.guideline      !== undefined || (options.colors.guideline  = "#EB2A18")
            options.colors.guidepoint     !== undefined || (options.colors.guidepoint = "#EB2A18")
            options.colors.guiderail      !== undefined || (options.colors.guiderail  = "#333333")
            options.guiderail             !== undefined || (options.guiderail = 0)
            options.size                  !== undefined || (options.size = {})
            options.size.radius           !== undefined || (options.size.radius = 30)
            options.size.segment          !== undefined || (options.size.segment = 4)
            options.transparency          !== undefined || (options.transparency = {})
            options.transparency.radius   !== undefined || (options.transparency.radius = 0.3)
            options.transparency.radius   !== undefined || (options.transparency.segment = 1.0)
            options.warning               !== undefined || (options.warning = "TOO FAST!<br/>GO BACK TO LAST POINT")
            options.limits                !== undefined || (options.limits = {})
            options.limits.distance       !== undefined || (options.limits.distance = 20)
            options.limits.restart        !== undefined || (options.limits.restart = 5)
            options.limits.bounds         !== undefined || (options.limits.bounds = {})
            options.limits.bounds.padding !== undefined || (options.limits.bounds.padding = true)
            options.limits.bounds.x       !== undefined || (options.limits.bounds.x = {})
            options.limits.bounds.x.min   !== undefined || (options.limits.bounds.x.min = 0)
            options.limits.bounds.x.max   !== undefined || (options.limits.bounds.x.max = 0)
            options.limits.bounds.y       !== undefined || (options.limits.bounds.y = {})
            options.limits.bounds.y.min   !== undefined || (options.limits.bounds.y.min = 0)
            options.limits.bounds.y.max   !== undefined || (options.limits.bounds.y.max = 0)
            options.debug                 !== undefined || (options.debug = false)

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
            this.options.padding = this.options.guiderail / (1.0 + (2.0 * this.options.guiderail))
            guideline_style = {
                "position": "absolute",
                "top":    (100.0 * this.options.padding) + "%",
                "bottom": (100.0 * this.options.padding) + "%",
                "left":   (100.0 * this.options.padding) + "%",
                "right":  (100.0 * this.options.padding) + "%",
                "border": "1px solid " + this.options.colors.guiderail,
                "color": this.options.colors.guiderail,
                "opacity": 0.5
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
                "margin-top": "-3px",
                "margin-left": "-3px",
                "z-index": "2",
            }

            // Begin guidepoint
            this.elements.guidepoint = $('<div id="' + this.options.ids.guidepoint + '"></div>')
            this.elements.guidepoint.css(guidepoint_style)
            this.elements.guidepoint.css({
                "background-color": this.options.colors.guidepoint,
            })
            this.elements.frame.append(this.elements.guidepoint)
            this.elements.guidepoint.hide()

            // Begin warning
            warning_style = {
                "position": "absolute",
                "left": "0",
                "top": "0",
                "width": "100%",
                "background-color": this.options.colors.guidepoint,
                "color": "#FFFFFF",
                "text-align": "center",
            }

            this.elements.warning = $('<div id="' + this.options.ids.warning + '"><b>' + this.options.warning + '</b></div>')
            this.elements.warning.css(warning_style)
            this.elements.frame.append(this.elements.warning)
            this.elements.warning.hide()
        }

        ContourSelector.prototype.check_boundaries = function(event) {
            var offset, point, bounds, width, height

            if (event == null) {
                return null
            }

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

            if (this.options.limits.bounds.padding) {
                width = this.elements.frame.width()
                height = this.elements.frame.height()

                bounds.x.min -= this.options.padding * width
                bounds.x.max -= this.options.padding * width
                bounds.y.min -= this.options.padding * height
                bounds.y.max -= this.options.padding * height
            }

            // Check lower boundaries
            point.x = Math.max(point.x, 0 - bounds.x.min)
            point.y = Math.max(point.y, 0 - bounds.y.min)
            // Check upper boundaries
            point.x = Math.min(point.x, this.elements.frame.width() - 1 + bounds.x.max)
            point.y = Math.min(point.y, this.elements.frame.height() - 1 + bounds.y.max)
            return point
        }

        ContourSelector.prototype.create_existing = function(existing) {
            invalid = false
            invalid = invalid || (existing.begin === undefined)
            invalid = invalid || (existing.segment === undefined)
            invalid = invalid || (existing.end === undefined)

            if (invalid) {
                existing = null
            }

            this.existing = existing
            this.draw_existing()
        }

        ContourSelector.prototype.draw_existing = function() {
            if (this.existing == null) {
                return
            }

            this.clear()

            this.points.begin   = this.existing.begin
            this.points.end     = this.existing.end
            this.points.segment = this.existing.segment

            this.elements.begin.show()
            this.elements.begin.css({
                "top":  (100.0 * this.points.begin.y) + "%",
                "left": (100.0 * this.points.begin.x) + "%",
            })

            this.elements.end.show()
            this.elements.end.css({
                "top":  (100.0 * this.points.end.y) + "%",
                "left": (100.0 * this.points.end.x) + "%",
            })

            for(var index = 0; index < this.points.segment.length; index++) {
                point = this.points.segment[index]
                this.draw_segment(point)
            }
        }

        ContourSelector.prototype.clear = function(keep_index) {

            keep_index !== undefined || (keep_index = null)

            if (keep_index < 0 || this.points.segment.length <= keep_index) {
                keep_index = null
            }

            this.elements.end.hide()
            this.elements.begin.hide()
            this.elements.guideline.hide()
            this.elements.guidepoint.hide()
            this.elements.warning.hide()

            this.ctx.clearRect(0, 0, this.elements.canvas.width(), this.elements.canvas.height());

            for(var index = 0; index < this.elements.segment.length; index++) {
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
            var width, height, last, diff_x, diff_y, dist

            index !== undefined || (index = this.points.segment.length - 1)

            if (index < 0) {
                return [-1, null, null]
            }

            width = this.elements.frame.width()
            height = this.elements.frame.height()

            last = this.points.segment[index]
            point = {
                "x": last.x * width,
                "y": last.y * height,
            }
            diff_x = this.points.cursor.x - point.x
            diff_y = this.points.cursor.y - point.y
            dist = Math.sqrt(diff_x ** 2 + diff_y ** 2)

            return [dist, last, point]
        }

        ContourSelector.prototype.start = function(event, index) {
            var width, height

            index !== undefined || (index = null)

            width = this.elements.frame.width()
            height = this.elements.frame.height()

            this.points.cursor = this.check_boundaries(event)

            if (index == null) {
                this.points.begin = {
                    "x": this.points.cursor.x / width,
                    "y": this.points.cursor.y / height,
                }
            } else {
                values = this.current_distance(index)
                distance = values[0]
                speedy = event.ctrlKey
                if (distance > this.options.limits.restart) {
                    return false
                }
            }

            this.clear(index)

            // Add starting point to the segment
            this.add_segment(event)

            // Show selectors, as needed
            this.elements.begin.show()
            this.elements.begin.css({
                "top":  (100.0 * this.points.begin.y) + "%",
                "left": (100.0 * this.points.begin.x) + "%",
            })

            // Set global cursor mode and prevent highlight selection
            $("body").css({
                "cursor": "crosshair"
            })

            document.onselectstart = function() {
                return false
            }

            return true
        }

        ContourSelector.prototype.add_segment = function(event) {
            var width, height, highlight, point

            width = this.elements.frame.width()
            height = this.elements.frame.height()

            // This functuion requires that points.cursor has already been updated
            highlight = event.shiftKey
            point = {
                "x": this.points.cursor.x / width,
                "y": this.points.cursor.y / height,
                "radius": {
                    "pixel":   this.options.size.radius,
                    "percent": this.options.size.radius / width,
                },
                "highlight": highlight
            }
            this.points.segment.push(point)

            return point
        }

        ContourSelector.prototype.draw_segment = function(point, radius_only) {
            var width, height, temp

            radius_only !== undefined || (radius_only = false)

            width = this.elements.frame.width()
            height = this.elements.frame.height()
            temp = {
                "x": point.x * width,
                "y": point.y * height,
                "radius": point.radius.percent * width,
            }

            // Draw radius on canvas
            this.ctx.beginPath()
            this.ctx.fillStyle = this.options.colors.radius
            this.ctx.ellipse(temp.x, temp.y, temp.radius, temp.radius, 0, 0, 2 * Math.PI)
            this.ctx.fill()

            if (radius_only) {
                return
            }

            if (point.highlight) {
                color = this.options.colors.highlight
            } else {
                color = this.options.colors.segment
            }

            size = this.options.size.segment
            size_half = Math.floor(size / 2.0)
            segment_style = {
                "top": (100.0 * point.y) + "%",
                "left": (100.0 * point.x) + "%",
                "width": size + "px",
                "height": size + "px",
                "position": "absolute",
                "border-radius": size + "px",
                "margin-top": "-" + size_half + "px",
                "margin-left": "-" + size_half + "px",
                "background-color": color,
                "opacity": this.options.transparency.segment,
                "z-index": "1",
            }

            // Begin segment center point
            segment = $('<div class="' + this.options.classes.segment + '"></div>')
            segment.css(segment_style)
            this.elements.frame.append(segment)
            this.elements.segment.push(segment)
        }

        ContourSelector.prototype.update_cursor = function(event) {
            var width, height, point
            var opacity, css_theta, css_rotate
            var highlight, point, size, size_half
            var segment, segment_style

            // Update the selector based on the current cursor location
            this.points.cursor = this.check_boundaries(event)

            width = this.elements.frame.width()
            height = this.elements.frame.height()
            values = this.current_distance()
            distance = values[0]
            last = values[1]
            point = values[2]

            speedy = event.ctrlKey
            if ( ! speedy && distance > this.options.limits.distance) {
                opacity = Math.max(0.25, Math.min(1.0, distance / 200.0))
                this.elements.guideline.css({
                    "top": (100.0 * last.y) + "%",
                    "left": (100.0 * last.x) + "%",
                    "width": distance + "px",
                    "opacity": opacity,
                })

                this.elements.guidepoint.css({
                    "top": (100.0 * last.y) + "%",
                    "left": (100.0 * last.x) + "%",
                    "opacity": Math.min(1.0, 2.0 * opacity),
                })

                css_theta = calculate_angle(point, this.points.cursor)
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

                this.elements.guideline.show()
                this.elements.guidepoint.show()
                this.elements.warning.show()

                return false
            }

            this.elements.guideline.hide()
            this.elements.guidepoint.hide()
            this.elements.warning.hide()

            // Add to the current segment and draw it
            point = this.add_segment(event)
            this.draw_segment(point)
        }

        ContourSelector.prototype.update_anchor = function(event) {
            var width, height, point
            var opacity, css_theta, css_rotate
            var highlight, point, size, size_half
            var segment, segment_style

            // Update the selector based on the current cursor location
            this.points.cursor = this.check_boundaries(event)

            width = this.elements.frame.width()
            height = this.elements.frame.height()

            closest_distance = Infinity
            closest_index = null
            for(var index = 0; index < this.points.segment.length; index++) {
                values = this.current_distance(index)
                distance = values[0]
                if (distance < closest_distance && distance < this.options.size.radius) {
                    closest_distance = distance
                    closest_index = index
                }
            }

            if (closest_index !== null) {
                closest_values = this.current_distance(closest_index)
                last = closest_values[1]
                this.elements.guidepoint.css({
                    "top": (100.0 * last.y) + "%",
                    "left": (100.0 * last.x) + "%",
                    "opacity": 1.0,
                })
                this.elements.guidepoint.show()
            } else {
                this.elements.guidepoint.hide()
            }

            return closest_index
        }

        ContourSelector.prototype.update_radius = function(radius) {
            this.options.size.radius = radius
            return this.options.size.radius
        }

        ContourSelector.prototype.finish = function(event) {
            var data, width, height

            this.points.cursor = this.check_boundaries(event)

            if (this.points.cursor == null) {
                this.clear()
                return
            }

            width = this.elements.frame.width()
            height = this.elements.frame.height()

            this.points.end = {
                "x": this.points.cursor.x / width,
                "y": this.points.cursor.y / height,
            }

            values = this.current_distance()
            distance = values[0]
            if (distance > this.options.limits.distance) {
                this.clear()
                return
            }

            // Add ending point to the segment
            this.add_segment(event)

            this.elements.end.show()
            this.elements.end.css({
                "top":  (100.0 * this.points.end.y) + "%",
                "left": (100.0 * this.points.end.x) + "%",
            })

            // Reset the global cursor mode and re-allow highlight selection
            $("body").css({
                "cursor": "default"
            })
            document.onselectstart = null
        }

        ContourSelector.prototype.get_selector_data = function(event) {
            var data

            // Create the selector data object
            if (this.points.segment.length == 0) {
                data = null
            } else {
                data = {
                    begin:   this.points.begin,
                    end:     this.points.end,
                    segment: this.points.segment,
                }
            }

            return data
        }

        ContourSelector.prototype.refresh = function(event) {
            data = this.get_selector_data(event)
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
            options.hotkeys                      !== undefined || (options.hotkeys = {})
            options.hotkeys.enabled              !== undefined || (options.hotkeys.enabled = true)
            options.hotkeys.delete               !== undefined || (options.hotkeys.delete = [8, 46, 75])
            options.hotkeys.exit                 !== undefined || (options.hotkeys.exit   = [27])
            options.hotkeys.zoom                 !== undefined || (options.hotkeys.zoom   = [90])
            options.guiderail                    !== undefined || (options.guiderail = 0)
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

            cta.delete_current(event)
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
                } else if (cta.state.mode == "free") {
                    cta.selector_anchor(event)
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

        ContourAnnotator.prototype.update_radius = function(radius) {
            if (this.cts === undefined) {
                radius = null
            } else {
                radius = this.cts.update_radius(radius)
            }
            return radius
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
            if(this.cts !== undefined) {
                this.cts.resize()
            }

            return this.refresh()
        }

        ContourAnnotator.prototype.zoom_start = function(event) {
            this.state.zoom = this.options.zoom.enabled
            this.selector_cancel(null)
            this.resize()
        }

        ContourAnnotator.prototype.zoom_finish = function(event) {
            this.state.zoom = false
            this.selector_cancel(null)
            this.resize()
        }

        ContourAnnotator.prototype.refresh = function() {
            if (this.cts === undefined) {
                data = null
            } else {
                data = this.cts.refresh()
            }

            if (this.options.callbacks.onchange != null) {
                this.options.callbacks.onchange(data)
            }

            return data
        }

        ContourAnnotator.prototype.delete_current = function(event) {
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
            // Update the ContourSelector
            this.cts.update_cursor(event)
        }

        ContourAnnotator.prototype.selector_anchor = function(event) {
            // Update the ContourSelector
            this.state.hover = this.cts.update_anchor(event)
        }

        ContourAnnotator.prototype.selector_cancel = function(event) {
            this.selector_finish(null)
        }

        ContourAnnotator.prototype.selector_finish = function(event) {
            // If we are in the "selector" mode, finish the ContourSelector and add the segment
            if (this.state.mode == "selector") {
                // Get the segment data by finishing the ContourSelector
                this.cts.finish(event)
                // Release the mode to "free"
                this.state.mode = "free"
                // Update the data
                this.refresh()
            }
        }

        ContourAnnotator.prototype.reset_existing = function(event) {
            this.cts.draw_existing()
            this.refresh()
        }

        return ContourAnnotator

    })()

}).call(this)