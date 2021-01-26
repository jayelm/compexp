"""
Visualize neurons as trees
"""

import json
import numpy as np


def make_treedata(
    layernames,
    contrs,
    tallies,
    by="feat_corr",
    root_collapsed=False,
    collapsed=True,
    root_name="resnet18",
    units=None,
    maxdepth=4,
    maxchildren=None,
    info_fn=None,
):
    # Loop in reverse order
    contrs_rev = contrs[::-1]
    tallies_rev = tallies[::-1]
    layernames_rev = layernames[::-1]
    tds = []
    if units is None:
        units = range(contrs_rev[0][by]["contr"][0].shape[0])
    for unit in units:
        td = _make_treedata_rec(
            unit,
            layernames_rev,
            contrs_rev,
            tallies_rev,
            root_name,
            1,
            maxdepth,
            by=by,
            collapsed=collapsed,
            maxchildren=maxchildren,
            info_fn=info_fn,
        )
        tds.append(td)
    if root_collapsed:
        k = "_children"
    else:
        k = "children"
    root = {"name": root_name, "parent": "null", k: tds}
    return root


def _make_treedata_rec(
    unit,
    layernames,
    contrs,
    tallies,
    parent_name,
    depth,
    maxdepth,
    parent_weight=None,
    by="feat_corr",
    collapsed=True,
    maxchildren=None,
    info_fn=None,
):
    # Loop in reverse order
    this_layername = layernames[0]
    this_contr, _ = contrs[0][by]["contr"]
    this_weight = contrs[0][by]["weight"]
    this_tally = tallies[0]
    this_name = f"{unit}-{this_tally[unit]['label']}"
    if parent_weight is not None:
        this_name = f"{this_name} ({parent_weight:.2f})"
    this = {
        "name": this_name,
        "parent": parent_name,
    }

    if info_fn is not None:
        this["info"] = info_fn(this_layername, unit)

    if this_contr is not None and depth < maxdepth:
        this_cs = np.where(this_contr[unit])[0]
        this_ws = this_weight[unit, this_cs]
        cws = sorted(zip(this_cs, this_ws), key=lambda cw: cw[1], reverse=True)
        if maxchildren is not None:
            cws = cws[:maxchildren]
        if collapsed:
            k = "_children"
        else:
            k = "children"
        this[k] = [
            _make_treedata_rec(
                u,
                layernames[1:],
                contrs[1:],
                tallies[1:],
                this_name,
                depth + 1,
                maxdepth,
                parent_weight=w,
                by=by,
                collapsed=collapsed,
                maxchildren=maxchildren,
                info_fn=info_fn,
            )
            for u, w in cws
        ]
    return this


EXAMPLE_TREEDATA = """
[
  {
    "name": "Top Level",
    "parent": "null",
    "children": [
      {
        "name": "Level 2: A",
        "parent": "Top Level",
        "children": [
          {
            "name": "Son of A",
            "parent": "Level 2: A"
          },
          {
            "name": "Daughter of A",
            "parent": "Level 2: A"
          }
        ]
      },
      {
        "name": "Level 2: B",
        "parent": "Top Level"
      }
    ]
  }
];
"""

TREESTYLE = r"""
<style>
    .node {
            cursor: pointer;
    }

    .node circle {
      fill: #fff;
      stroke: steelblue;
      stroke-width: 3px;
    }

    .node text {
      font: 12px sans-serif;
    }

    .link {
      fill: none;
      stroke: #ccc;
      stroke-width: 2px;
    }
    /* Tooltip container */
    div.tooltip {
        position: absolute;
        text-align: left;
        width: 500px;
        height: 80px;
        padding: 8px;
        font: 10px sans-serif;
        pointer-events: auto;
    }
</style>
"""

TREESCRIPT = r"""
<script>

// ************** Generate the tree diagram     *****************
var margin = {top: 20, right: 120, bottom: 20, left: 120};

var width = 1960 - margin.right - margin.left;

// For the full 365
// var height = 5000 - margin.top - margin.bottom;
// For 36
var height = 2000 - margin.top - margin.bottom;

var i = 0,
    duration = 750;

var root = treeData;
root.x0 = height / 2;
root.y0 = 0;

var tree = d3.layout.tree()
    .size([height, width]);

var diagonal = d3.svg.diagonal()
    .projection(function(d) { return [d.y, d.x]; });

var svg = d3.select("body").append("svg")
    .attr("width", width + margin.right + margin.left)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// Toggle children on click.
var update = function(source) {

  var click = function(d) {
    if (d.children) {
      d._children = d.children;
      d.children = null;
    } else {
      d.children = d._children;
      d._children = null;
    }
    update(d);
  }

  var info = d3.select('body')
      .append('div')
      .attr('class', 'tooltip')
      .html('')
      .style('display', 'none')
      .on('mouseover', function(d, i) {
        info.transition().duration(0).style('display', 'block');
      })
      .on('mouseout', function(d, i) {
        info.style('display', 'none');
      });

  // Compute the new tree layout.
  var nodes = tree.nodes(root).reverse(),
      links = tree.links(nodes);

  // Normalize for fixed-depth.
  nodes.forEach(function(d) { d.y = d.depth * 180; });

  // Update the nodes…
  var node = svg.selectAll("g.node")
      .data(nodes, function(d) { return d.id || (d.id = ++i); });

  // Enter any new nodes at the parent's previous position.
  var nodeEnter = node.enter().append("g")
      .attr("class", "node")
      .attr("transform", function(d) { return "translate(" + source.y0 + "," + source.x0 + ")"; })
      .on("click", click)
      .on("mouseover", function(d) {
          info.transition().duration(0);
          var g = d3.select(this); // The node
          // The class is used to remove the additional text later
          info
             .html(d.info)
             .style("display", 'block')
             .style("opacity", "1")
             .style("left", (d3.event.pageX + 20 + 'px'))
             .style("top", (d3.event.pageY + 'px'));
      })
      .on("mouseout", function() {
          // Remove the info text on mouse out.
          info.transition()
            .delay(500)
            .style('display', 'none');
      });
    ;

  nodeEnter.append("circle")
      .attr("r", 1e-6)
      .style("fill", function(d) { return d._children ? "lightsteelblue" : "#fff"; });

  nodeEnter.append("text")
      .attr("x", function(d) { return d.children || d._children ? -13 : 13; })
      .attr("dy", ".35em")
      .attr("text-anchor", function(d) { return d.children || d._children ? "end" : "start"; })
      .text(function(d) { return d.name; })
      .style("fill-opacity", 1e-6);

  // Transition nodes to their new position.
  var nodeUpdate = node.transition()
      .duration(duration)
      .attr("transform", function(d) { return "translate(" + d.y + "," + d.x + ")"; });

  nodeUpdate.select("circle")
      .attr("r", 10)
      .style("fill", function(d) { return d._children ? "lightsteelblue" : "#fff"; });

  nodeUpdate.select("text")
      .style("fill-opacity", 1);

  // Transition exiting nodes to the parent's new position.
  var nodeExit = node.exit().transition()
      .duration(duration)
      .attr("transform", function(d) { return "translate(" + source.y + "," + source.x + ")"; })
      .remove();

  nodeExit.select("circle")
      .attr("r", 1e-6);

  nodeExit.select("text")
      .style("fill-opacity", 1e-6);

  // Update the links…
  var link = svg.selectAll("path.link")
      .data(links, function(d) { return d.target.id; });

  // Enter any new links at the parent's previous position.
  link.enter().insert("path", "g")
      .attr("class", "link")
      .attr("d", function(d) {
        var o = {x: source.x0, y: source.y0};
        return diagonal({source: o, target: o});
      });

  // Transition links to their new position.
  link.transition()
      .duration(duration)
      .attr("d", diagonal);

  // Transition exiting nodes to the parent's new position.
  link.exit().transition()
      .duration(duration)
      .attr("d", function(d) {
        var o = {x: source.x, y: source.y};
        return diagonal({source: o, target: o});
      })
      .remove();

  // Stash the old positions for transition.
  nodes.forEach(function(d) {
    d.x0 = d.x;
    d.y0 = d.y;
  });
}

update(root);

d3.select(self.frameElement).style("height", "500px");

</script>
"""
