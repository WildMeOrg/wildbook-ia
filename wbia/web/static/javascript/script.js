function submit_cookie(name, value) {
  $.ajax({
      url: "/ajax/cookie/?name=" + name + "&value=" + value,
      success: function(response){
        // console.log("COOKIE: " + response);
      },
      error: function(response){
        // console.log("COOKIE: " + response);
      }
  });
}

Array.min = function( array ){
    return Math.min.apply( Math, array );
};

Array.max = function( array ){
    return Math.max.apply( Math, array );
};

$.fn.getIndex = function(filter){
    var index = $(this).parent().children(filter).index( $(this) );
    return index;
}

function contains(arr, obj) {
    var i = arr.length;
    while (i--) {
       if (arr[i] === obj) {
           return true;
       }
    }
    return false;
}

function findValueIndexFromArray(value, array)
{
    var index = array.indexOf(value);
    return index
}

function removeValueFromArray(value, array)
{
    var index = findValueIndexFromArray(value, array);
    removeIndexFromArray(index, array)
    return index
}

function removeIndexFromArray(index, array)
{
    if (index > -1) {
        array.splice(index, 1);
    }
}

function randomHexColor()
{
  // 16777215 base 10 === ffffff base 16
  return '#'+Math.floor(Math.random()*16777215).toString(16);
}

function syntaxHighlight(json) {
    json = JSON.stringify(reorder(json), undefined, 4);
    json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
        var cls = 'highlight-number';
        if (/^"/.test(match)) {
            if (/:$/.test(match)) {
                cls = 'highlight-key';
            } else {
                cls = 'highlight-string';
            }
        } else if (/true|false/.test(match)) {
            cls = 'highlight-boolean';
        } else if (/null/.test(match)) {
            cls = 'highlight-null';
        }
        return '<span class="' + cls + '">' + match + '</span>';
    });
}

function reorder(obj){
    if(typeof obj !== 'object')
        return obj
    var temp = {};
    var keys = [];
    for(var key in obj)
        keys.push(key);
    keys.sort();
    for(var index in keys)
        temp[keys[index]] = reorder(obj[keys[index]]);
    return temp;
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
