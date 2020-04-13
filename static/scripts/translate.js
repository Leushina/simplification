$(function() {
  //Translate text with flask route
  $("#translate").on("click", function(e) {
    e.preventDefault();
    var translateVal = document.getElementById("text-to-translate").value;
    var translateRequest = { 'text': translateVal }
    if (translateVal !== "") {
      $.ajax({
        url: '/simplify',
        method: 'POST',
        headers: {
            'Content-Type':'application/json'
        },
        dataType: 'json',
        data: JSON.stringify(translateRequest),
        success: function(data) {
        console.log(data)
        for (var i = 0; i < data.simplifications.length; i++) {
          document.getElementById("translation-result").textContent = data.simplifications[i].translation;}
        }
      });
    };
  });
  })