function inside_polygon(point, vs) {
    // ray-casting algorithm based on
    // http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html

    var x = point[0], y = point[1];

    var inside = false;
    for (var i = 0, j = vs.length - 1; i < vs.length; j = i++) {
        var xi = vs[i][0], yi = vs[i][1];
        var xj = vs[j][0], yj = vs[j][1];

        var intersect = ((yi > y) != (yj > y))
            && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }

    return inside;
};

function render_polygon(color_polycon, polygon, color_text, text, modx_text, mody_text) {
  modx_text !== undefined || (modx_text = 0.0);
  mody_text !== undefined || (mody_text = 0.0);

  canv = $(canvas)
  width = canv.outerWidth()
  height = canv.outerHeight()

  ctx.fillStyle = color_polycon;
  ctx.beginPath();

  minx = Infinity
  maxx = -1.0 * Infinity
  miny = Infinity
  maxy = -1.0 * Infinity
  for (var index = 0; index < polygon.length; index++) {
    point = polygon[index]

    minx = Math.min(point[0], minx)
    maxx = Math.max(point[0], maxx)
    miny = Math.min(point[1], miny)
    maxy = Math.max(point[1], maxy)

    point[0] *= width
    point[1] *= height

    if (index == 0) {
      ctx.moveTo(point[0], point[1]);
    } else {
      ctx.lineTo(point[0], point[1]);
    }
  }

  ctx.closePath();
  ctx.fill();

  point = [(minx + maxx) * 0.5 + modx_text, (miny + maxy) * 0.5 + mody_text]

  point[0] *= width
  point[1] *= height

  ctx.fillStyle = color_text;
  ctx.fillText(text, point[0], point[1]);
}

function render_canvas(option, hover) {
  option !== undefined || (option = 1);
  hover !== undefined  || (hover = null);

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.font="bold 20px Arial";
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'center';

  // Values
  center_innerx = 0.16
  center_outerx = 0.28
  center_innery = 0.18
  center_outery = 0.28

  vert_midx = -0.06
  vert_midy = 0.16
  vert_endx = 0.06;

  hori_midx = 0.20
  hori_midy = -0.05
  hori_endy = 0.18

  // Offsets
  outerx_offset = (1.00 - center_outerx) / 2.0
  outery_offset = (1.00 - center_outery) / 2.0

  innerx_offset = (1.00 - center_innerx) / 2.0
  innery_offset = (1.00 - center_innery) / 2.0

  vert_midx_offset = outerx_offset - vert_midx
  vert_midy_offset = outery_offset - vert_midy

  vert_endx_offset = vert_midx_offset - vert_endx
  vert_endy_offset = 0.0

  hori_midx_offset = outerx_offset - hori_midx
  hori_midy_offset = outery_offset - hori_midy

  hori_endx_offset = 0.0
  hori_endy_offset = hori_midy_offset - hori_endy

  // Center
  polygon1 = [
    [innerx_offset,       outery_offset],
    [1.0 - innerx_offset, outery_offset],
    [1.0 - outerx_offset, innery_offset],
    [1.0 - outerx_offset, 1.0 - innery_offset],
    [1.0 - innerx_offset, 1.0 - outery_offset],
    [innerx_offset,       1.0 - outery_offset],
    [outerx_offset,       1.0 - innery_offset],
    [outerx_offset,       innery_offset],
  ]

  // Vert Mid Upper
  polygon2 = [
    [innerx_offset,          outery_offset],
    [1.0 - innerx_offset,    outery_offset],
    [1.0 - vert_midx_offset, vert_midy_offset],
    [vert_midx_offset,       vert_midy_offset],
  ]

  // Vert Mid Lower
  polygon3 = [
    [innerx_offset,          1.0 - outery_offset],
    [1.0 - innerx_offset,    1.0 - outery_offset],
    [1.0 - vert_midx_offset, 1.0 - vert_midy_offset],
    [vert_midx_offset,       1.0 - vert_midy_offset],
  ]

  // Hori Mid Left
  polygon4 = [
    [(outerx_offset),    (innery_offset)],
    [(outerx_offset),    (1.0 - innery_offset)],
    [(hori_midx_offset), (1.0 - hori_midy_offset)],
    [(hori_midx_offset), (hori_midy_offset)],
  ]

  // Hori Mid Right
  polygon5 = [
    [(1.0 - outerx_offset),    (innery_offset)],
    [(1.0 - outerx_offset),    (1.0 - innery_offset)],
    [(1.0 - hori_midx_offset), (1.0 - hori_midy_offset)],
    [(1.0 - hori_midx_offset), (hori_midy_offset)],
  ]

  // Vert End Upper
  polygon6 = [
    [1.0 - vert_midx_offset, vert_midy_offset],
    [vert_midx_offset,       vert_midy_offset],
    [vert_endx_offset,       0.0],
    [1.0 - vert_endx_offset, 0.0],
  ]

  // Vert End Lower
  polygon7 = [
    [1.0 - vert_midx_offset, 1.0 - vert_midy_offset],
    [vert_midx_offset,       1.0 - vert_midy_offset],
    [vert_endx_offset,       1.0],
    [1.0 - vert_endx_offset, 1.0],
  ]

  // Hori End Left
  polygon8 = [
    [hori_midx_offset, 1.0 - hori_midy_offset],
    [hori_midx_offset, hori_midy_offset],
    [0.0,              hori_endy_offset],
    [0.0,              1.0 - hori_endy_offset],
  ]

  // Hori End Right
  polygon9 = [
    [1.0 - hori_midx_offset, 1.0 - hori_midy_offset],
    [1.0 - hori_midx_offset, hori_midy_offset],
    [1.0,                    hori_endy_offset],
    [1.0,                    1.0 - hori_endy_offset],
  ]

  // Triple Upper-Left
  polygon10 = [
    [innerx_offset,    outery_offset],
    [vert_midx_offset, vert_midy_offset],
    [vert_endx_offset, 0.0],
    [hori_midx_offset, hori_midy_offset],
    [outerx_offset,    innery_offset],
  ]

  // Triple Bottom-Left
  polygon11 = [
    [innerx_offset,    1.0 - outery_offset],
    [vert_midx_offset, 1.0 - vert_midy_offset],
    [vert_endx_offset, 1.0],
    [hori_midx_offset, 1.0 - hori_midy_offset],
    [outerx_offset,    1.0 - innery_offset],
  ]

  // Triple Upper-Right
  polygon12 = [
    [1.0 - innerx_offset,    outery_offset],
    [1.0 - vert_midx_offset, vert_midy_offset],
    [1.0 - vert_endx_offset, 0.0],
    [1.0 - hori_midx_offset, hori_midy_offset],
    [1.0 - outerx_offset,    innery_offset],
  ]

  // Triple Bottom-Right
  polygon13 = [
    [1.0 - innerx_offset,    1.0 - outery_offset],
    [1.0 - vert_midx_offset, 1.0 - vert_midy_offset],
    [1.0 - vert_endx_offset, 1.0],
    [1.0 - hori_midx_offset, 1.0 - hori_midy_offset],
    [1.0 - outerx_offset,    1.0 - innery_offset],
  ]

  // Corner Upper-Left
  polygon14 = [
    [0.0,              0.0],
    [0.0,              hori_endy_offset],
    [hori_midx_offset, hori_midy_offset],
    [vert_endx_offset, 0.0],
  ]

  // Corner Upper-Right
  polygon15 = [
    [1.0,                    0.0],
    [1.0,                    hori_endy_offset],
    [1.0 - hori_midx_offset, hori_midy_offset],
    [1.0 - vert_endx_offset, 0.0],
  ]

  // Corner Bottom-Left
  polygon16 = [
    [0.0,              1.0],
    [0.0,              1.0 - hori_endy_offset],
    [hori_midx_offset, 1.0 - hori_midy_offset],
    [vert_endx_offset, 1.0],
  ]

  // Corner Bottom-Right
  polygon17 = [
    [1.0,                    1.0],
    [1.0,                    1.0 - hori_endy_offset],
    [1.0 - hori_midx_offset, 1.0 - hori_midy_offset],
    [1.0 - vert_endx_offset, 1.0],
  ]

  polygons = [
    polygon1,
    polygon2,
    polygon3,
    polygon4,
    polygon5,
    polygon6,
    polygon7,
    polygon8,
    polygon9,
    polygon10,
    polygon11,
    polygon12,
    polygon13,
    polygon14,
    polygon15,
    polygon16,
    polygon17,
  ]

  if (option == 1) {
    texts = [
      'FRONT',
      'U-F', 'D-F',
      'F-R', 'F-L',
      'UP', 'DOWN',
      'RIGHT', 'LEFT',
      'U-F-R', 'D-F-R', 'U-F-L', 'D-F-L',
      'U-R', 'U-L', 'D-R', 'D-L',
    ]
  } else if (option == 2) {
    texts = [
      'BACK',
      'U-B', 'D-B',
      'B-L', 'B-R',
      'UP', 'DOWN',
      'LEFT', 'RIGHT',
      'U-B-L', 'D-B-L', 'U-B-R', 'D-B-R',
      'U-L', 'U-R', 'D-L', 'D-R',
    ]
  } else if (option == 3) {
    texts = [
      'LEFT',
      'U-L', 'D-L',
      'F-L', 'B-L',
      'UP', 'DOWN',
      'FRONT', 'BACK',
      'U-F-L', 'D-F-L', 'U-B-L', 'D-B-L',
      'U-F', 'U-B', 'D-F', 'D-B',
    ]
  } else if (option == 4) {
    texts = [
      'RIGHT',
      'U-R', 'D-R',
      'B-R', 'F-R',
      'UP', 'DOWN',
      'BACK', 'FRONT',
      'U-B-R', 'D-B-R', 'U-F-R', 'D-R-F',
      'U-B', 'U-F', 'D-B', 'D-F',
    ]
  } else if (option == 5) {
    texts = [
      'UP',
      'U-L', 'U-R',
      'U-B', 'U-F',
      'LEFT', 'RIGHT',
      'BACK', 'FRONT',
      'U-B-L', 'U-B-R', 'U-F-L', 'U-F-R',
      'B-L', 'F-L', 'B-R', 'F-R',
    ]
  } else if (option == 6) {
    texts = [
      'DOWN',
      'D-R', 'D-L',
      'D-B', 'D-F',
      'RIGHT', 'LEFT',
      'BACK', 'FRONT',
      'D-B-R', 'D-B-L', 'D-F-R', 'D-F-L',
      'B-R', 'F-R', 'B-L', 'F-L',
    ]
  } else {
    console.log('INVALID OPTION ' + option)
    return
  }

  colors = [
    [(hover == null || hover == 0  ? '#C9302C' : '#F3C8C7'), '#FFF'],
    [(hover == null || hover == 1  ? '#5CB85C' : '#C9E8C9'), '#FFF'],
    [(hover == null || hover == 2  ? '#5CB85C' : '#C9E8C9'), '#FFF'],
    [(hover == null || hover == 3  ? '#5CB85C' : '#C9E8C9'), '#FFF'],
    [(hover == null || hover == 4  ? '#5CB85C' : '#C9E8C9'), '#FFF'],
    [(hover == null || hover == 5  ? '#337AB7' : '#D0E2F2'), '#FFF'],
    [(hover == null || hover == 6  ? '#337AB7' : '#D0E2F2'), '#FFF'],
    [(hover == null || hover == 7  ? '#337AB7' : '#D0E2F2'), '#FFF'],
    [(hover == null || hover == 8  ? '#337AB7' : '#D0E2F2'), '#FFF'],
    [(hover == null || hover == 9  ? '#EC971F' : '#fAE3C2'), '#FFF'],
    [(hover == null || hover == 10 ? '#EC971F' : '#fAE3C2'), '#FFF'],
    [(hover == null || hover == 11 ? '#EC971F' : '#fAE3C2'), '#FFF'],
    [(hover == null || hover == 12 ? '#EC971F' : '#fAE3C2'), '#FFF'],
    [(hover == null || hover == 13 ? '#5CB85C' : '#C9E8C9'), '#FFF'],
    [(hover == null || hover == 14 ? '#5CB85C' : '#C9E8C9'), '#FFF'],
    [(hover == null || hover == 15 ? '#5CB85C' : '#C9E8C9'), '#FFF'],
    [(hover == null || hover == 16 ? '#5CB85C' : '#C9E8C9'), '#FFF'],
  ]

  render_polygon(colors[0][0],  polygons[0],  colors[0][1],  texts[0])
  render_polygon(colors[1][0],  polygons[1],  colors[1][1],  texts[1])
  render_polygon(colors[2][0],  polygons[2],  colors[2][1],  texts[2])
  render_polygon(colors[3][0],  polygons[3],  colors[3][1],  texts[3])
  render_polygon(colors[4][0],  polygons[4],  colors[4][1],  texts[4])
  render_polygon(colors[5][0],  polygons[5],  colors[5][1],  texts[5])
  render_polygon(colors[6][0],  polygons[6],  colors[6][1],  texts[6])
  render_polygon(colors[7][0],  polygons[7],  colors[7][1],  texts[7])
  render_polygon(colors[8][0],  polygons[8],  colors[8][1],  texts[8])
  render_polygon(colors[9][0],  polygons[9],  colors[9][1],  texts[9], 0.05, 0.05)
  render_polygon(colors[10][0], polygons[10], colors[10][1], texts[10], 0.05, -0.05)
  render_polygon(colors[11][0], polygons[11], colors[11][1], texts[11], -0.05, 0.05)
  render_polygon(colors[12][0], polygons[12], colors[12][1], texts[12], -0.05, -0.05)
  render_polygon(colors[13][0], polygons[13], colors[13][1], texts[13], -0.02, -0.04)
  render_polygon(colors[14][0], polygons[14], colors[14][1], texts[14], 0.02, -0.04)
  render_polygon(colors[15][0], polygons[15], colors[15][1], texts[15], -0.02, 0.04)
  render_polygon(colors[16][0], polygons[16], colors[16][1], texts[16], 0.02, 0.04)

  text = (hover != null && 0 <= hover && hover < texts.length ? texts[hover] : null)
  return polygons, text
}

function add_species()
{
  value = $('input[name="species-add"]').val()
  console.log(value);

  $('select[name="viewpoint-species"]')
     .append($("<option></option>")
     .attr("value", value)
     .text(value));
  $('select[name="viewpoint-species"] option[value="' + value + '"]').prop('selected', true)
}

var hotkeys_disabled = false;

$('#species-add').on('shown.bs.modal', function () {
  $('input[name="species-add"]').val('')
  hotkeys_disabled = true;
});

$('#species-add').on('hidden.bs.modal', function () {
  hotkeys_disabled = false;
});


$(window).keydown(function(event) {
  key = event.which;
  console.log(key);
  console.log('disabled ' + hotkeys_disabled);

  if( ! hotkeys_disabled)
  {
    if(key == 13)
    {
      // Enter key pressed, submit form as accept
      $('input#turk-submit-accept').click();
    }
    else if (key == 17) {
      // Ctrl pressed
      $('.ia-detection-hotkey').show();
    }
    else if(key == 32)
    {
      // Space key pressed, submit form as delete
      $('input#turk-submit-delete').click();
    }
    else if (key == 67) {
      // C pressed
      var element = $('select[name="viewpoint-species"]')
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
    }
    else if(key == 76)
    {
      // L key pressed, submit form as left
      $('input#turk-submit-left').click();
    }
    else if(key == 82)
    {
      // R key pressed, submit form as right
      $('input#turk-submit-right').click();
    }
    else if(key == 80)
    {
      // P key pressed, follow previous link
      $('a#turk-previous')[0].click();
    }
  }
});