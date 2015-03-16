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