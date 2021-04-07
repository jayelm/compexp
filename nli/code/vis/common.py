"""
Common HTML things
"""


def unquote(x):
    return x.replace('"', r"&quot;").replace("'", r"&apos;")


ACCORDION = """
<div class="accordion mb-2" id="a{accordion_id}">{body}</div>
"""

ACCORDION_MEMBER = """
  <div class="card">
    <div class="card-header" id="a{id}-header">
      <h2 class="mb-0">
        <button class="btn btn-link" type="button" data-toggle="collapse" data-target="#a{id}">
        {title}
        </button>
      </h2>
    </div>

    <div id="a{id}" class="collapse" data-parent="#a{accordion_id}">
      <div class="card-body">
          {body}
      </div>
    </div>
  </div>
"""

HTML_PREFIX = r"""
<!DOCTYPE html>
<html>
<head>
    <title>Report</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css" integrity="sha384-Vkoo8x4CGsO3+Hhxv8T/Q5PaXtkKtu6ug5TOeNV6gBiFeWPGFN9MuhOf23Q9Ifjh" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.4.1.slim.min.js" integrity="sha384-J6qa4849blE2+poT4WnyKhv5vZF5SrPo0iEjwBvKU7imGFAV0wwj1yYfoRSJoZ+n" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" crossorigin="anonymous"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js" integrity="sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6" crossorigin="anonymous"></script>
<style>
.currentsort {
  background-color: black;
  font-weight: bold;
}
.sort-up::after {
  content: " - (up)";
}
.sort-down::after {
  content: " - (down)";
}
.sentence {
    line-height: 3em;
}
.neuron:hover {
    background-color: yellow;
}
.unit {
    margin-top: 1em;
    margin-bottom: 1em;
}
.card-deck-b {
    padding-left: 1em;
    paddding-right: 1em;
}
.word {
    margin-right: 10px;
    position: relative;
}
.contradiction {
    color: darkred;
    font-weight: bold;
}
.entailment {
    color: darkgreen;
    font-weight: bold;
}
.neutral {
    font-weight: bold;
}
.word::before {
    position: absolute;
    top: 0.3em;
    left: 0;
    content: attr(data-tag);
    text-align: center;
    color: #aaa;
    font-size: 0.8em;
}
.tooltip-inner {
    background: none;
    color: #212529;
    text-align: left;
}
.fname {
    font-weight: bold;
}
.correct {
    background-color: rgba(0, 255, 0, 0.1);
}
.incorrect {
    background-color: rgba(255, 0, 0, 0.1);
}
.neuron {
    margin-right: 10px;
    cursor: pointer;
}
.sentence {
    background: none;
}
.cm-section {
    margin-top: 5px;
}
</style>
</head>

<body>
<nav class="navbar navbar-light justify-content-center mt-4">
    <form class="form-inline">
        <button type="button" class="btn btn-primary btn-sm mr-3 sortby currentsort sort-up" data-sortby="unit">Unit</button>
        <button type="button" class="btn btn-primary btn-sm mr-3 sortby" data-sortby="iou">IoU</button>
        <button type="button" class="btn btn-primary btn-sm mr-3 sortby" data-sortby="label">Label</button>
        <button type="button" class="btn btn-primary btn-sm mr-3 sortby" data-sortby="category">Category</button>
        <button type="button" class="btn btn-primary btn-sm mr-3 sortby" data-sortby="correct">Correct</button>
        <button type="button" class="btn btn-primary btn-sm mr-3 sortby" data-sortby="entail">Entailment</button>
        <button type="button" class="btn btn-primary btn-sm mr-3 sortby" data-sortby="neutral">Neutral</button>
        <button type="button" class="btn btn-primary btn-sm mr-3 sortby" data-sortby="contradiction">Contradiction</button>
        <button type="button" class="btn btn-primary btn-sm mr-3 sortby" data-sortby="snli-entropy">Entropy</button>
    </form>
</nav>
<div class="card-deck-b">
"""

HTML_SUFFIX = r"""
</div>
<script>
$(document).ready(function() {
    $('.neuron').mouseover(function() {
        var neuron = parseInt($(this).data('neuron'));
        $('.word').each(function() {
            var actn = 'act-' + neuron;
            if ($(this).data(actn)) {
                var val = parseFloat($(this).data(actn));
                if (val > 0) {
                    var alpha = Math.min(1.0, Math.max(0.0, val));
                    var bg = 'rgba(0, 0, 255, ' + alpha + ')';
                } else {
                    var alpha = Math.min(1.0, Math.max(0.0, -val));
                    var bg = 'rgba(255, 0, 0, ' + alpha + ')';
                }
                $(this).css('background-color', bg);
            }
        });
    });
    $('.word').each(function() {
        if ($(this).data('act')) {
            var val = parseFloat($(this).data('act'));
            if (val > 0) {
                var alpha = Math.min(1.0, Math.max(0.0, val));
                var bg = 'rgba(0, 0, 255, ' + alpha + ')';
            } else {
                var alpha = Math.min(1.0, Math.max(0.0, -val));
                var bg = 'rgba(255, 0, 0, ' + alpha + ')';
            }
            $(this).css('background-color', bg);
        }
    });
    $('.sortby').click(function() {
        if ($(this).hasClass('currentsort')) {
            if ($(this).hasClass('sort-up')) {
                // switch to negative sort
                var dir = -1;
                $(this).removeClass('sort-up');
                $(this).addClass('sort-down');
            } else {
                // switch to positive sort
                var dir = 1;
                $(this).removeClass('sort-down');
                $(this).addClass('sort-up');
            }
        } else {
            // default to positive sort
            var dir = 1;
            $('.sortby').removeClass('currentsort');
            $('.sortby').removeClass('sort-up');
            $('.sortby').removeClass('sort-down');
            $(this).addClass('currentsort');
            $(this).addClass('sort-up');
        }
        var attr = $(this).data('sortby');
        $('.card-deck-b .card.unit').sort(function(a, b) {
            return dir * ($(a).data(attr) > $(b).data(attr) ? 1 : -1);
        }).appendTo(".card-deck-b");
    });
    $('[data-toggle="tooltip"]').tooltip()
});
</script>

</body>
</html>
"""


CARD_HTML = """
<div class="card unit" data-unit="{unit}" data-iou="{iou}" data-label="{label}" data-entail="{entail}" data-neutral="{neutral}" data-contra="{contra}">
  <div class="card-body">
    <h5 class="card-title">{title}</h5>
    <h6 class="card-subtitle mb-2 text-muted">{subtitle}</h6>
    {items}
  </div>
</div>
"""

SCARD_HTML = """
<div class="card unit" data-unit="{unit}" data-iou="{iou}" data-label="{label}" data-category="{category}" data-entail="{entail}" data-neutral="{neutral}" data-contra="{contra}" data-snli-entropy="{snli_entropy}">
  <div class="card-body">
    <h5 class="card-title">{title}</h5>
    <h6 class="card-subtitle mb-2 text-muted">{subtitle}</h6>
    {items}
  </div>
</div>
"""

CM_TABLE = """
<div class="cm">
<h5 class="text-muted cmtitle">{}</h5>
<table class="table card-table cmtable">
  <thead>
    <tr>
      <th scope="col"></th>
      <th scope="col">PRED: entail</th>
      <th scope="col">PRED: neutral</th>
      <th scope="col">PRED: contra </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">GT: entail</th>
      <td>{}</td>
      <td>{}</td>
      <td>{}</td>
    </tr>
    <tr>
      <th scope="row">GT: neutral</th>
      <td>{}</td>
      <td>{}</td>
      <td>{}</td>
    </tr>
    <tr>
      <th scope="row">GT: contra</th>
      <td>{}</td>
      <td>{}</td>
      <td>{}</td>
    </tr>
  </tbody>
</table>
</div>
"""
