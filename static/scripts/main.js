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
        document.getElementById("translation-result").textContent = data.simplifications
        }
      });
    };
  });

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
  $("#search").on("click", function(e) {
    e.preventDefault();
    var searchVal = document.getElementById("text-to-search").value;
    var searchRequest = { 'text': searchVal }
    if (searchVal !== "") {
      $.ajax({
        url: '/search',
        method: 'POST',
        headers: {
            'Content-Type':'application/json'
        },
        dataType: 'json',
        data: JSON.stringify(searchRequest),
        success: function(data) {
        for (var i = 0; i < data.search_results.length; i++) {
            idx = "result " + (i+1).toString();
            x = document.getElementById(idx);
            x.textContent = data.search_results[i];
            if (x.classList.contains("invisible")) {
                x.classList.remove("invisible");
                x.classList.add("visible");
                }
            }
        x = document.getElementById("search-container");
        x.classList.remove("col-sm-3");
        x.classList.add("col-sm-6");

        x = document.getElementById("simplify-container");
        x.classList.remove("col-md-7");
        x.classList.add("col-md-4");

         x = document.getElementById("article to display");
        if (x.classList.contains("visible")) {
            x.classList.remove("visible");
            x.innerHTML = "";
            x.classList.add("invisible");}
        }
      });
    };
  });

  var reply_click = function()
{
    console.log("clicked, id "+ this.id+", text"+this.innerHTML);
    var articleRequest = {'idx': this.id }
     $.ajax({
        url: '/show_article',
        method: 'POST',
        headers: {
            'Content-Type':'application/json'
        },
        dataType: 'json',
        data: JSON.stringify(articleRequest),
        success: function(data) {
            for (var i = 0; i < 5; i++) {
                idx = "result " + (i+1).toString();
                x = document.getElementById(idx);
                x.innerHTML = "";
                if (x.classList.contains("visible")) {
                    x.classList.remove("visible");
                    x.classList.add("invisible");
                }
                }
            x = document.getElementById("article to display");
            x.innerHTML = data.article;
            x.classList.remove("invisible");
            x.classList.add("visible");
            }
      });

}
  document.getElementById('result 1').onclick = reply_click;
  document.getElementById('result 2').onclick = reply_click;
  document.getElementById('result 3').onclick = reply_click;
  document.getElementById('result 4').onclick = reply_click;
  document.getElementById('result 5').onclick = reply_click;
  })