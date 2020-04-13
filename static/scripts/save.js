$(function() {
    $("#save").on("click", function(e) {
    e.preventDefault();
    var inputVal = document.getElementById("text-to-translate").value;
    var simpleVal = document.getElementById("translation-result").value;
    var evaluateVal = 0;
    if (document.getElementById('radios1').checked) {
        evaluateVal = 1;
    };
    var translateRequest = { 'input_txt': inputVal, 'simple_txt':simpleVal, 'evaluate': evaluateVal}
    console.log(translateRequest)
    $.ajax({
        url: '/save',
        method: 'POST',
        headers: {
            'Content-Type':'application/json'
        },
        dataType: 'json',
        data: JSON.stringify(translateRequest),
        success: function(data) {
            window.alert("Your data is successfully saved");
            document.getElementById("text-to-translate").textContent = "";
            document.getElementById("translation-result").textContent = "";
        }
      });
  });
})