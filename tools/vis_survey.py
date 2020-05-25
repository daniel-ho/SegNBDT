import os
from jinja2 import Template
import sys


files = sys.argv[1:]

template = Template(
"""
<html>
  <head>
    <title>Figure Browser</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.17/d3.min.js"></script>
    <script
      src="https://code.jquery.com/jquery-3.5.1.min.js"
      integrity="sha256-9/aliU8dGd2tb6OSsuzixeV4y/faTqgFtohetphbbj0="
      crossorigin="anonymous"></script>
    <script>
var index = -1;
var files = [
    {% for file in files %}
        '{{ file }}',
    {% endfor %}
];

function select(i, dontPushHistory) {
  i = parseInt(i);
  index = i;
  $('#curr').html(i);

  $('svg').remove();
  $('main').load(files[i - 1], function() {
    console.log('Loaded ' + i);
  })

  if (i <= 1) {
    $('#prev').hide();
  } else {
    var url = '?id=' + (i-1);
    $('#prev').show().attr('i', i-1).attr('href', '?id=' + (i-1));
    $('#prev span').html(i - 1);
  }

  if (i >= files.length) {
    $('#next').hide();
  } else {
    var url = '?id=' + (i+1);
    $('#next').show().attr('i', i+1).attr('href', url);
    $('#next span').html(i + 1);
  }

  if (!dontPushHistory) {
    history.pushState('data', '', '?id=' + i);
  }
}

function onLoadSelect() {
  const urlParams = new URLSearchParams(window.location.search);
  const i = urlParams.get('id') || 0;
  select(i, true);
}

$(document).ready(function() {
  $('#total').html(files.length);
  onLoadSelect();

  $('#next').on('click', function(e) {
      select($('#next').attr('i'));
      e.preventDefault();
  });

  $('#prev').on('click', function(e) {
      select($('#prev').attr('i'));
      e.preventDefault();
  });

  $(window).on('popstate', onLoadSelect);
});
    </script>
  </head>
  <body>
    <header>
      <p>
        <span>Figure ID: <b id="curr">-1</b> of <span id="total">-1</span></span>
        <a href="#" id="prev">prev (<span>-1</span>)</a>
        <a href="#" id="next">next (<span>-1</span>)</a>
      </p>
    </header>
    <main>
    </main>
  </body>
</html>

""")

with open("index.html", "w") as f:
    f.write(template.render(files=files))
