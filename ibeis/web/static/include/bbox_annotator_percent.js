(function() {
  var BBoxSelector;

  BBoxSelector = (function() {
    function BBoxSelector(frame, options) {
      var options;

      options              || (options = {});
      options.border       || (options.border = {});
      options.border.color || (options.border.color = "#7FFF7F");
      options.border.width || (options.border.width = 2);
      options.mode         || (options.mode = "diagonal");
      options.debug        || (options.debug = false);

      // Global attributes
      this.options = options;

      this.points = {};
      this.elements = {};
      this.current_stage = 0;  // Only used when the diagonal selector is used
      this.debug = [];

      // Create objects
      this.create_selector_elements(frame);

      // Set (update) selector mode
      this.update_mode(this.options.mode)
    }

    BBoxSelector.prototype.create_selector_elements = function(frame) {
      var selector_style, debug_style;

      // Define the image frame as an element
      this.elements.frame = frame;

      // Default style inherited by both selectors
      selector_style = {
        "border":           this.options.border.width + "px dotted " + this.options.border.color,
        "position":         "absolute",
        "transform-origin": "0% 0%",
      }

      // Create rectangle selector object
      this.elements.rectangle = $('<div class="ia-bbox-selector-rectangle"></div>');
      this.elements.rectangle.css(selector_style);
      this.elements.frame.append(this.elements.rectangle);
      this.elements.rectangle.hide();

      // Create diagonal selector object
      this.elements.diagonal = $('<div class="ia-bbox-selector-diagonal"></div>');
      this.elements.diagonal.css(selector_style);
      this.elements.frame.append(this.elements.diagonal);

      // Create the debug objects for visual purposes
      if(this.options.debug) {
        debug_style = {
          "left":          "0.0",
          "top":           "0.0",
          "width":         "10px",
          "height":        "10px",
          "position":      "absolute",
          "border-radius": "10px",
          "border":        "1px solid #fff",
          "margin-top":    "-6px",
          "margin-left":   "-6px",
        }

        colors = ["red", "green", "blue"]
        for (index = 0; index < colors.length; index++) {
          this.debug.push( $("<div></div>") );
          this.debug[index].css(debug_style);
          this.debug[index].css({
            "background-color": colors[index],
          })
          this.elements.frame.append(this.debug);
        }
      }
    }

    BBoxSelector.prototype.update_mode = function(proposed_mode) {
      // The only valid modes are "rectangle" and "diagonal"
      switch (proposed_mode) {
        case "rectangle":
        case "diagonal":
          this.options.mode = proposed_mode;
          break;
        default:
          console.log("[BBoxSelector] Invalid selection mode provided: " + proposed_mode)
      }

      // Return the active mode
      return this.options.mode
    }

    BBoxSelector.prototype.check_boundaries = function(pageX, pageY) {
      var offset, point;

      offset = this.elements.frame.offset()
      point = {
        x: pageX - offset.left,
        y: pageY - offset.top,
      };
      // Check lower boundaries
      point.x = Math.max(point.x, 0);
      point.y = Math.max(point.y, 0);
      // Check upper boundaries
      point.x = Math.min(point.x, this.elements.frame.width() - 1);
      point.y = Math.min(point.y, this.elements.frame.height() - 1);
      return point;
    };

    BBoxSelector.prototype.start = function(pageX, pageY) {
      this.points.origin = this.check_boundaries(pageX, pageY);
      this.points.middle = {
        x: NaN,
        y: NaN,
      }
      this.points.cursor = this.points.origin;

      this.current_stage = 0;

      // Show selectors, as needed
      if(this.options.mode === "rectangle")
      {
        this.elements.rectangle.show();
        this.elements.diagonal.hide();
        this.elements.rectangle.css({
          "opacity": "0.0",
        });
      }
      else
      {
        this.elements.rectangle.hide();
        this.elements.diagonal.show();
        this.elements.diagonal.css({
          "opacity": "0.0",
          "width":   "0px",
        });
      }

      // Set global cursor mode and prevent highlight selection
      $("body").css({
        "cursor": "crosshair"
      });
      document.onselectstart = function() {
        return false;
      };

      // Don't refresh until we get new coordinates from the BBoxAnnotator
      // this.refresh();
    };

    BBoxSelector.prototype.stage = function(pageX, pageY) {
      // Used only for a multi-staging selection (i.e. diagonal)
      if(this.options.mode == "diagonal" && this.current_stage == 0)
      {
        this.current_stage = 1 // We only want to stage once
        this.elements.rectangle.show();
      }
      this.refresh();
    };


    BBoxSelector.prototype.finish = function() {
      var data;

      data = this.get_selector_data();

      // Hide selectors
      this.elements.rectangle.hide();
      this.elements.diagonal.hide();

      // Reset the global cursor mode and re-allow highlight selection
      $("body").css({
        "cursor": "default"
      });
      document.onselectstart = null;

      return data;
    };

    BBoxSelector.prototype.update_cursor = function(pageX, pageY) {
      // Update the selector based on the current cursor location
      this.points.cursor = this.check_boundaries(pageX, pageY);
      if(this.current_stage != 1)
      {
        this.points.middle = this.points.cursor
      }

      this.refresh();
    };

    BBoxSelector.prototype.calculate_angle = function(pointa, pointb) {
      var xa, ya, xb, yb, x1, x2, y1, y2, x, y, angle;

      // Get the points and coordinates of extrema
      xa = pointa.x
      ya = pointa.y
      xb = pointb.x
      yb = pointb.y
      x1 = Math.min(xa, xb);
      x2 = Math.max(xa, xb);
      y1 = Math.min(ya, yb);
      y2 = Math.max(ya, yb);
      x = x2 - x1
      y = y2 - y1

      // calculate the angle, preserving sign after arctangent
      if(xa < xb && ya < yb) {
        angle  = Math.atan2(y, x);
      }
      else if(xa < xb && ya >= yb) {
        angle  = Math.atan2(y, x);
        angle *= -1.0
      }
      else if(xa >= xb && ya < yb) {
        angle  = Math.atan2(x, y);
        angle += 0.5 * Math.PI
      }
      else if(xa >= xb && ya >= yb) {
        angle  = Math.atan2(y, x);
        angle += Math.PI
      }

      // Normalize the angle to be between 0 and 2*PI
      while(angle < 0) {
        angle += 2.0 * Math.PI
      }
      while(angle > 2.0 * Math.PI) {
        angle -= 2.0 * Math.PI
      }
      return angle;
    }

    BBoxSelector.prototype.get_selector_data = function() {
      var width, height, data, diff_x, diff_y;

      // Create the selector data object
      data   = {
        pixels:     {
          origin:   this.points.origin,
          middle:   this.points.middle,
          cursor:  this.points.cursor,
        },
        percent:    {},
        angles:     {},
        metadata: {
          id:       null,
        },
      };

      // Calculate the raw pixel locations and angles
      data.pixels.xtl     = Math.min(data.pixels.origin.x, data.pixels.middle.x);
      data.pixels.ytl     = Math.min(data.pixels.origin.y, data.pixels.middle.y);
      data.pixels.xbr     = Math.max(data.pixels.origin.x, data.pixels.middle.x);
      data.pixels.ybr     = Math.max(data.pixels.origin.y, data.pixels.middle.y);
      data.pixels.width   = data.pixels.xbr - data.pixels.xtl;
      data.pixels.height  = data.pixels.ybr - data.pixels.ytl;

      data.pixels.hypo    = Math.sqrt(data.pixels.width ** 2 + data.pixels.height ** 2);
      diff_x              = data.pixels.middle.x - data.pixels.cursor.x
      diff_y              = data.pixels.middle.y - data.pixels.cursor.y
      data.pixels.right   = Math.sqrt(diff_x ** 2 + diff_y ** 2);

      data.angles.origin  = this.calculate_angle(data.pixels.origin, data.pixels.middle)
      data.angles.middle  = this.calculate_angle(data.pixels.middle, data.pixels.cursor)
      data.angles.right   = data.angles.origin - Math.PI * 0.5
      data.angles.theta   = data.angles.origin

      // Calculate the left, top, width, and height in pixel space
      if(this.options.mode == "rectangle") {
        data.pixels.left  = data.pixels.xtl;
        data.pixels.top   = data.pixels.ytl;
        data.angles.theta = 0.0;
      }
      else {
        var diff_angle, diff_len, position, center;

        if(this.current_stage > 0) {
          // Calculate the perpendicular angle and offset from the first segment
          diff_angle = data.angles.middle - data.angles.origin
          diff_len = Math.sin(diff_angle) * data.pixels.right
          data.pixels.adjacent = Math.abs(diff_len)
          data.pixels.correction = {
            x: Math.cos(data.angles.right) * data.pixels.adjacent,
            y: Math.sin(data.angles.right) * data.pixels.adjacent,
          }

          if(diff_len > 0) {
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

        if(this.debug.length > 0) {
          this.debug[0].css({
            "left": data.pixels.xtl + "px",
            "top":  data.pixels.ytl + "px",
          });
          this.debug[1].css({
            "left": centers.hypo.x + "px",
            "top":  centers.hypo.y + "px",
          });
          this.debug[2].css({
            "left": centers.rect.x + "px",
            "top":  centers.rect.y + "px",
          });
        }

        // Normalize the pixel locations based on center rotation
        data.pixels.left   = data.pixels.xtl + centers.offset.x;
        data.pixels.top    = data.pixels.ytl + centers.offset.y;
        data.pixels.width  = this.elements.rectangle.width();
        data.pixels.height = this.elements.rectangle.height();
      }

      // Get the current width and height of the image to calculate percentages
      width  = this.elements.frame.width();
      height = this.elements.frame.height();

      // Calculate the final values in percentage space
      data.percent.left   = 100.0 * data.pixels.left / width;
      data.percent.top    = 100.0 * data.pixels.top / height;
      data.percent.width  = 100.0 * data.pixels.width / width;
      data.percent.height = 100.0 * data.pixels.height / height;

      // Normalize the theta angle range to be between 0 and 2*PI
      while(data.angles.theta < 0) {
        data.angles.theta += 2.0 * Math.PI
      }
      while(data.angles.theta > 2.0 * Math.PI) {
        data.angles.theta -= 2.0 * Math.PI
      }

      return data;
    };

    BBoxSelector.prototype.refresh = function() {
      var data, css_theta, css_rotate;

      data = this.get_selector_data();

      if(this.options.mode == "rectangle") {
        css_theta = data.angles.theta

        // Update the appearance of the rectangle selection using percentages
        this.elements.rectangle.css({
          "left":    data.percent.left   + "%",
          "top":     data.percent.top    + "%",
          "width":   data.percent.width  + "%",
          "height":  data.percent.height + "%",
          "opacity": "1.0",
        });
      }
      else {
        css_theta = data.angles.origin

        if(this.current_stage == 0) {
          // First stage of the diagonal selection: drawing the hypotenuse line
          this.elements.diagonal.css({
            "left":    data.pixels.origin.x + "px",
            "top":     data.pixels.origin.y + "px",
            "width":   data.pixels.hypo     + "px",
            "opacity": "1.0",
          })
          this.elements.rectangle.css({
            "left":   data.pixels.origin.x + "px",
            "top":    data.pixels.origin.y + "px",
            "width":  data.pixels.hypo     + "px",
            "height": 0 + "px",
          });
        }
        else {
          // Second stage of the diagonal selection: expanding the rectangle
          this.elements.diagonal.css({
            "opacity": "0.5",
          })
          this.elements.rectangle.css({
            "left":   (data.pixels.origin.x + data.pixels.correction.x) + "px",
            "top":    (data.pixels.origin.y + data.pixels.correction.y) + "px",
            "width":  data.pixels.hypo + "px",
            "height": 2.0 * data.pixels.adjacent + "px",
          });
        }
      }

      // Rotate the boxes based on the appropriate values
      css_rotate = {
        "transform":         "rotate(" + css_theta + "rad)",
        "-o-transform":      "rotate(" + css_theta + "rad)",
        "-ms-transform":     "rotate(" + css_theta + "rad)",
        "-moz-transform":    "rotate(" + css_theta + "rad)",
        "-webkit-transform": "rotate(" + css_theta + "rad)",
      }

      this.elements.rectangle.css(css_rotate);
      this.elements.diagonal.css(css_rotate);
    };

    return BBoxSelector;
  })();









  this.BBoxAnnotator = (function() {
    function BBoxAnnotator(url, options) {
      var options;

      options                 || (options = {});
      options.ids             || (options.ids = {});
      options.ids.container   || (options.ids.container  = "#ia-bbox-annotator-container");
      options.classes         || (options.classes = {});
      options.classes.bbox    || (options.classes.bbox   = ".ia-bbox-annotator-bbox");
      options.classes.label   || (options.classes.label  = ".ia-bbox-annotator-bbox-label");
      options.classes.close   || (options.classes.close  = ".ia-bbox-annotator-bbox-close");
      options.classes.resize  || (options.classes.resize = ".ia-bbox-annotator-bbox-resize");
      options.classes.rotate  || (options.classes.rotate = ".ia-bbox-annotator-bbox-rotate");
      options.colors          || (options.colors = {});
      options.colors.default  || (options.colors.default = "#7FFF7F");
      options.colors.hover    || (options.colors.hover = "#F0AD4E");
      options.colors.target   || (options.colors.target = "#EB2A18");
      options.colors.interest || (options.colors.interest = "#2E63FF");
      options.border          || (options.border = {});
      options.border.color    || (options.border.color = options.colors.default);
      options.border.width    || (options.border.width = 2);
      options.mode            || (options.mode = "diagonal");
      options.debug           || (options.debug = false);

      // Global attributes
      this.options = options

      this.entries = [];
      this.elements = {
        entries: [],
      }

      // Keep track of the current state of the annotator
      this.state = {
        status: "free",
        hover:  null,
      }

      this.create_annotator_elements(url);
    }

    BBoxAnnotator.prototype.create_annotator_elements = function(url) {
      var bba;

      bba = this;

      // Define the container
      this.elements.container = $(this.options.ids.container);
      this.elements.container.css({
        "width":        "100%",
        "max-width":    "1200px",
        "margin-left":  "auto",
        "margin-right": "auto",
      });

      // Create the image frame
      this.elements.frame = $('<div class="ia-bbox-annotator-frame"></div>');
      this.elements.frame.css({
        "position":          "relative",
        "width":             "100%",
        "height":            "100%",
        "border":            "#333333 solid 1px",
        "margin":            "0px auto",
        "background-size":   "100% auto",
        "background-repeat": "no-repeat",
        "overflow":          "hidden",
        "cursor":            "crosshair",
      });
      this.elements.container.append(this.elements.frame);

      // Create a new Javascript Image object and load the src provided by the user
      this.elements.image = new Image();
      this.elements.image.src = url;

      // Create events to load the rest of the data, but only if the image loads
      this.elements.image.onload = function() {
        // Set the background image of the image frame to be the iamge src
        bba.elements.frame.css({
          "background-image": "url(\"" + this.src + "\")",
        });

        // We can now create the selector object
        bba.bbs = new BBoxSelector(bba.elements.frame, bba.options);

        // This is the last stage of the constructor's initialization
        bba.register_global_event_bindings();
      };

      // If the image failed to load, add a message to the console.
      this.elements.image.onerror = function() {
        console.log.text("[BBoxAnnotator] Invalid image URL provided: " + this.url);
      };
    }

    BBoxAnnotator.prototype.resize = function() {
      var w1, h1, w2, h2;

      // Get the proportions of the image and
      w1 = this.elements.image.width
      h1 = this.elements.image.height
      w2 = this.elements.frame.width();
      h2 = (w2 / w1) * h1;

      this.elements.container.css({
        "height": h2 + "px",
      });
    }

    BBoxAnnotator.prototype.update_mode = function(proposed_mode) {
      this.options.mode = this.bbs.update_mode(proposed_mode)
    };

    BBoxAnnotator.prototype.refresh = function() {
      if (this.options.onchange) {
        return this.options.onchange(this.entries);
      }
    };

    BBoxAnnotator.prototype.delete_entry = function(index) {
        // Detatch the bbox HTML elements at index
        this.elements.entries[index].bbox.detach();

        // Slice out the JSON entry and HTML elements from their respective list
        this.entries.splice(index, 1);
        this.elements.entries.splice(index, 1);

        // Refresh the display
        this.refresh();
    }

    BBoxAnnotator.prototype.update_hover = function(index) {
        console.log('HOVER ' + index)
        this.state.hover = index;
    }


    BBoxAnnotator.prototype.add_entry = function(entry) {
      var element;

      element = {};
      console.log('[BBoxAnnotator] Add entry: ' + entry);

      // Create the bbox container
      element.bbox = $('<div class="' + this.options.classes.bbox + '"></div>');
      element.bbox.css({
        "position":              "absolute",
        "top":                   entry.percent.top + "%",
        "left":                  entry.percent.left + "%",
        "width":                 entry.percent.width + "%",
        "height":                entry.percent.height + "%",
        "border":                this.options.border.width + "px solid " + this.options.colors.default,
        "color":                 "#FFFFFF",
        "font-family":           "monospace",
        "font-size":             "small",
      });
      // Apply orientation to the bbox
      css_rotate = {
        "transform":             "rotate(" + entry.angles.theta + "rad)",
        "-o-transform":          "rotate(" + entry.angles.theta + "rad)",
        "-ms-transform":         "rotate(" + entry.angles.theta + "rad)",
        "-moz-transform":        "rotate(" + entry.angles.theta + "rad)",
        "-webkit-transform":     "rotate(" + entry.angles.theta + "rad)",
      }
      element.bbox.css(css_rotate);
      this.elements.frame.append(element.bbox);

      // Add the label to the bbox to denote the ID
      element.label = $('<div class="' + this.options.classes.label + '"></div>');
      element.label.css({
        "overflow":              "visible",
        "display":               "inline-block",
        "background-color":      this.options.colors.default,
        "opacity":               "0.8",
        "color":                 "#333333",
        "padding":               "1px 3px",
      });
      element.bbox.append(element.label);

      // Add the close button to the bbox
      element.close = $('<div class="' + this.options.classes.close + '"></div>');
      element.close.css({
        "position":              "absolute",
        "top":                   "5px",
        "right":                 "5px",
        "width":                 "20px",
        "height":                "0",
        "border":                "2px solid #FFFFFF",
        "margin-left":           "-10px",
        "padding":               "16px 0 0 0",
        "color":                 "#FFFFFF",
        "background-color":      "#003300",
        "cursor":                "pointer",
        "user-select":           "none",
        "text-align":            "center",
        "overflow":              "visible",
        "border-radius":         "18px",
        "-moz-border-radius":    "18px",
        "-webkit-border-radius": "18px",
        "-moz-user-select":      "none",
        "-webkit-user-select":   "none",
      });
      element.bbox.append(element.close);

      // Add the X to the close button, which we do not need to keep track of
      temp = $("<div></div>");
      temp.html("&#215;").css({
        "position":              "absolute",
        "top":                   "-2px",
        "left":                  "0",
        "width":                 "16px",
        "display":               "block",
        "text-align":            "center",
        "font-size":             "20px",
        "line-height":           "20px",
      });
      element.close.append(temp);

      // Add the rotation handle to the bbox
      element.rotate = $('<div class="' + this.options.classes.rotate + '"></div>');
      element.rotate.css({
        "position":              "absolute",
        "top":                   "5px",
        "right":                 "50%",
        "width":                 "10px",
        "height":                "10px",
        "border":                "1px solid #333333",
        "margin-left":           "-10px",
        "background-color":      this.options.colors.default,
        "cursor":                "pointer",
        "border-radius":         "10px",
        "-moz-border-radius":    "10px",
        "-webkit-border-radius": "10px",
      });
      element.bbox.append(element.rotate);

      // Register entry event bindings for this new element
      this.register_entry_event_bindings(element);

      // Add the entry JSON and also the jQuery elements
      this.entries.push(entry);
      this.elements.entries.push(element);

      // Refresh the view now that the new bbox has been added
      this.refresh();
    };

    BBoxAnnotator.prototype.register_entry_event_bindings = function(element) {
      var bba;

      bba = this;

      // Register events for when the box is hovered and unhovered
      element.bbox.hover((function(e) {
        var index;

        index = $(this).prevAll(bba.options.classes.bbox).length;
        bba.update_hover(index)
      }), (function(e) {
        bba.update_hover(null)
      }));

      // Register event for when the close button is selected for a specific bbox
      element.close.mouseup(function(e) {
        var index;

        // Find the parent bbox of the close button and the bbox's index
        parent = $(this).parent(bba.options.classes.bbox)
        index = parent.prevAll(bba.options.classes.bbox).length;

        // Delete the entry from the annotator
        bba.delete_entry(index);
      });
    };


///////////////
///////////////
///////////////
///////////////

    BBoxAnnotator.prototype.selector_start = function(pageX, pageY) {
        this.bbs.start(pageX, pageY);
        this.state.status = "hold";
        this.state.adding = true;
        this.state.editing = false;
    };


    BBoxAnnotator.prototype.selector_finish = function() {
      var data;
      switch (this.state.status) {
        case "input":
          data = this.bbs.finish();
          this.add_entry(data);
          this.state.status = "free";
      }
      this.state.adding = false;
      this.state.moving = true;
      return true;
    };

    BBoxAnnotator.prototype.update_style_hover = function(box_element, text_box, resizable_button, rotate_button, close_button)
    {
        box_element.css("outline", "none");
        box_element.css("border-color", this.options.colors.hover);
        box_element.css("opacity", "1.0");
        box_element.css("cursor", "move");
        box_element.css("background-color", "rgba(0, 0, 0, 0.2)");
        box_element.addClass("ia-annotated-bounding-box-active");
        box_element.removeClass("ia-annotated-bounding-box-target");
        text_box.css("background-color", this.options.colors.hover);
        text_box.css("opacity", "1.0");
        text_box.css("color", "black");

        resizable_button.show();
        rotate_button.show();
        close_button.show();
    }

    BBoxAnnotator.prototype.update_style_editing = function(box_element, text_box, resizable_button, rotate_button, close_button)
    {
        box_element.css("outline", "none");
        box_element.css("border-color", this.options.colors.default);
        box_element.css("opacity", "0.8");
        box_element.css("cursor", "crosshair");
        box_element.css("background-color", "rgba(0, 0, 0, 0.0)");
        box_element.css("background-color", "rgba(0, 0, 0, 0.0)");
        box_element.removeClass("ia-annotated-bounding-box-active");
        box_element.removeClass("ia-annotated-bounding-box-target");
        text_box.css("background-color", this.options.colors.default);
        text_box.css("opacity", "0.8");
        text_box.css("color", "black");

        resizable_button.show();
        rotate_button.hide();
        close_button.hide();
    }

    BBoxAnnotator.prototype.update_style_target = function(box_element, text_box, resizable_button, rotate_button, close_button)
    {
        box_element.css("outline", "1000px solid rgba(0, 0, 0, 0.2)");
        box_element.css("border-color", this.options.colors.target);
        box_element.css("opacity", "1.0");
        box_element.css("cursor", "cell");
        box_element.css("background-color", "rgba(0, 0, 0, 0.0)");
        box_element.addClass("ia-annotated-bounding-box-target");
        text_box.css("background-color", this.options.colors.target);
        text_box.css("opacity", "1.0");
        text_box.css("color", "white");
        $(".ia-annotated-bounding-box").not(".ia-annotated-bounding-box-target").css("visibility", "hidden");

        rotate_button.hide();
        resizable_button.hide();
        close_button.hide();
    }

    BBoxAnnotator.prototype.update_theta = function(entry, theta)
    {
        function mod(x, n) {
          // Javascript % is not modulus, it is remainder (wtf?)
          return ((x % n) + n) % n;
        }
        entry["theta"] = mod(theta, 2.0 * Math.PI);
        this.refresh();
    }





    BBoxAnnotator.prototype.register_global_event_bindings = function(selector, options) {
      var bba;

      bba = this;

      this.hit_menuitem = false;


      $(window).bind("resize", function(){
        bba.resize();
      });


      $(window).keydown(function(e) {
        console.log('WINDOW KEYDOWN')

        if(e.which === 27 || e.which === 75 || e.which === 8)
        {
          // Esc
          bba.state.adding = false;
          bba.state.editing = false;
        }

        switch (bba.state.status) {
          case "hold":
            if (e.which === 27 || e.which === 75 || e.which === 8) {
              data = bba.bbs.finish();
              bba.state.adding = false;
              bba.state.status = "free";
            }
            break;
          case "free":
            var active_box, index;
            active_box = $(".ia-annotated-bounding-box-active");

            if(active_box.length > 0)
            {
              index = active_box.prevAll(".ia-annotated-bounding-box").length;
              move = 1.0
              if(e.shiftKey) {
                move *= 10.0;
              }
              img_width = parseInt(bba.elements.frame.css("width"));
              img_height = parseInt(bba.elements.frame.css("height"));
              if(e.which === 27 || e.which === 75 || e.which === 8)
              {
                if(bba.state.targeting)
                {
                  bba.state.targeting = false;
                  if(bba.state.hovering)
                  {
                    $(".ia-annotated-bounding-box-target").trigger("mouseenter");
                  }
                  else
                  {
                    $(".ia-annotated-bounding-box-target").trigger("mouseleave");
                  }
                }
                else
                {
                  // Delete
                  active_box.detach();
                  bba.entries.splice(index, 1);
                  bba.refresh();
                  bba.state.editing = false;
                }
              }

              if(e.which === 66)
              {
                // Send to front
                temp = active_box.detach();
                temp.prependTo(bba.elements.frame);
                entry = bba.entries.splice(index, 1);
                bba.entries.unshift(entry[0]);
                bba.refresh();
              }

              if(e.which === 37)
              {
                // Left
                proposed = parseFloat(active_box.css("left")) - move;
                proposed = Math.max(0.0, proposed);
                active_box.css("left", (100.0 * proposed / img_width) + "%");
                entry = bba.entries[index];
                entry.left = proposed;
                bba.refresh();
              }
              else if(e.which === 38)
              {
                // Up
                proposed = parseFloat(active_box.css("top")) - move;
                proposed = Math.max(0.0, proposed);
                active_box.css("top", (100.0 * proposed / img_height) + "%");
                entry = bba.entries[index];
                entry.top = proposed;
                bba.refresh();
                    e.preventDefault();
                return false;
              }
              else if(e.which === 39)
              {
                // Right
                proposed = parseFloat(active_box.css("left")) + move;
                proposed = Math.min(img_width - parseFloat(active_box.css("width")) - 1, proposed);
                active_box.css("left", (100.0 * proposed / img_width) + "%");
                entry = bba.entries[index];
                entry.left = proposed;
                bba.refresh();
              }
              else if(e.which === 40)
              {
                // Down
                proposed = parseFloat(active_box.css("top")) + move;
                proposed = Math.min(img_height - parseFloat(active_box.css("height")) - 1, proposed);
                active_box.css("top", (100.0 * proposed / img_height) + "%");
                entry = bba.entries[index];
                entry.top  = proposed;
                bba.refresh();
                e.preventDefault();
                return false;
              }
            }
        }
      });

      this.elements.container.mousedown(function(e) {
        console.log('CONTAINER MOUSEDOWN')

        if( bba.bbs.options.mode == "rectangle" || (! bba.bbs.options.mode == "rectangle" && bba.bbs.current_stage == 1))
        {
          $(".ia-annotated-bounding-box").css("opacity", "1.00");
        }

        if (!bba.hit_menuitem) {
          switch (bba.state.status) {
            case "free":
            case "input":
              if (bba.state.status === "input") {
                bba.selector_finish();
              }
              if (e.which === 1) {
                bba.selector_start(e.pageX, e.pageY);
              }
              else if (e.which === 3) {
                // pass
              }
              break;
            case "hold":
              bba.bbs.update_cursor(e.pageX, e.pageY);
              if(bba.bbs.options.mode == "rectangle")
              {
                bba.state.status = "input";
                bba.selector_finish();
              }
              else
              {
                if(bba.bbs.current_stage == 0)
                {
                  bba.bbs.stage(e.pageX, e.pageY)
                }
                else
                {
                  bba.bbs.current_stage = 2
                  // bba.state.adding = false
                  bba.state.status = "input";
                  bba.selector_finish();
                }
              }
          }
        }
        else if (bba.hit_menuitem)
        {
            switch (bba.state.status) {
                case "free":
                case "input":
                    if (e.which === 1) {
                      // pass
                    }
                    else if (e.which === 3) {
                        bba.state.editing = true;
                        $(".ia-label-text-box").css("opacity", "0.0");
                    }
            }
        }
        else if(!bba.state.editing)
        {
          // bba.hit_menuitem = false;
        }
        return true;
      });

      this.elements.container.mouseup(function(e) {
        console.log('CONTAINER MOUSEUP')

        if(bba.bbs.options.mode == "rectangle" || (! bba.bbs.options.mode == "rectangle" && bba.bbs.current_stage == 2))
        {
          $(".ia-annotated-bounding-sbox").css("opacity", "1.00");
          $(".ia-label-text-box").css("opacity", "0.8");
        }

        switch (bba.state.status) {
          case "hold":
            bba.bbs.update_cursor(e.pageX, e.pageY);
            if(bba.bbs.options.mode == "rectangle")
            {
              bba.state.adding = false;
              bba.state.status = "input";
              bba.selector_finish();
            }
            else
            {
              if(bba.bbs.current_stage == 2)
              {
                bba.state.adding = false;
                bba.state.status = "input";
                bba.selector_finish();
              }
            }
        }
        return true;
      });

      $(window).mousemove(function(e) {
        // console.log('WINDOW MOUSEMOVE')

        if(bba.state.editing)
        {
          bba.state.moving = true;
        }

        switch (bba.state.status) {
          case "hold":
            bba.bbs.update_cursor(e.pageX, e.pageY);
            $(".ia-annotated-bounding-box").css("opacity", "0.25");
            $(".ia-label-text-box").css("opacity", "0.0");
        }
        return true;
      });


      this.resize();
    };




    return BBoxAnnotator;

  })();

}).call(this);
