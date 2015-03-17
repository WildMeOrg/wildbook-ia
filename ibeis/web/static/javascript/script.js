function submit_cookie(name, value) {
  $.ajax({
      url: "/ajax/cookie?name=" + name + "&value=" + value,
      success: function(response){
        console.log("COOKIE: " + response);
      },
      error: function(response){
        console.log("COOKIE: " + response);
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
