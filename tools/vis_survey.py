import os
from jinja2 import Template
import sys
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--a', nargs='*')
parser.add_argument('--b', nargs='*')
args = parser.parse_args()

# attempts to match stem
stem_to_path = {}
for a in args.a:
    stem_to_path[Path(a).stem] = [a]

for b in args.b:
    stem = Path(b).stem
    if stem in stem_to_path:
        stem_to_path[stem].append(b)

files = [value for value in stem_to_path.values() if len(value) == 2]

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
        {a: '{{ file[0] }}', b: '{{ file[1] }}'},
    {% endfor %}
];

function select(i, dontPushHistory) {
  i = parseInt(i);
  index = i;
  $('#curr').html(i);

  var file = files[i - 1];
  $('div#figure-a').html('<img src="' + file['a'] + '"><p>Goal: Classify center pixel.</p><p>Prediction: Car</p>');
  $('div#figure-b').load(file['b'], function() {
    console.log('Loaded ' + i);

    if (window.onload) {
        window.onload();
    }
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
  const i = urlParams.get('id') || 1;
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
    <style>
.comparison {
    display:flex;
    flex-direction:row;
}
.explanation {
    flex:1;
}
.explanation + .explanation {
margin-left:5em;
}
h1 {
    margin:0;
    padding:0;
}
    </style>
  </head>
  <body>
    <header>
      <p>
        <span>Figure ID: <b id="curr">-1</b> of <span id="total">-1</span></span>
        <a href="#" id="prev">prev (<span>-1</span>)</a>
        <a href="#" id="next">next (<span>-1</span>)</a>
      </p>
    </header>
    <div class="comparison">
        <div class="explanation">
        <h1>Explanation A</h1>
            <div id="figure-a"></div>
        </div>
        <div class="explanation">
            <h1>Explanation B</h1>
            <div id="figure-b"></div>
        </div>
    </div>
  </body>
</html>

""")

with open("index.html", "w") as f:
    f.write(template.render(files=files))
