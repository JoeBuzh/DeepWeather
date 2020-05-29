'use strict';

var d3 = require('d3');

angular.module('app').
   directive('lineChart', LineChartDirective)
   
function LineChartDirective($parse) {
	return {
        scope: {
            values: '='
        },
        template: '<svg width="390" height="165"></svg>',
        restrict: 'E',
        controller: () => {},
        bindToController: true,
        controllerAs: 'viz',
        link: function (scope, element, attrs, ctrl) {
            // Bring in the Bubbles class
            // Create a Bubbles visualization, targeting the SVG element from the template
            var visualization = new LineChart(element.find('svg')[0]);
            // Watch for any changes to the values array, and when it changes, re-render the chart
            scope.$watchCollection(function () {
				return ctrl.values;
			}, function () {
				//console.log(attrs.name);
				visualization.render(ctrl.values ? ctrl.values : [], attrs.name);
			});
            scope.$on('$destroy', () => {
                // If we have anything to clean up when the scope gets destroyed
                visualization.destroy();
            });
        }
    };
}



   
function LineChart(target) {
    this.target = target;
}

function type(d, _, columns) {
    d.time = parseTime(d.time);
    for (var i = 1, n = columns.length, c; i < n; ++i) d[c = columns[i]] = +d[c];
    return d;
}
var parseTime = d3.timeParse("%Y-%m-%d %H:%M:%S");
 
// Does the work of drawing the visualization in the target area
LineChart.prototype.render = function (data, name) {
	if (data.values === undefined) {
		return;
	}
	
	// data: {key: Array}, legend: array
	var legend = data.legend;
	var data = data.values;
	
	var values = [];
	for(var i in data) {
		values = values.concat(data[i]);
	}
	
	var svg = d3.select(this.target);
	var margin = {top: 20, right: 20, bottom: 20, left: 30},
	    width = +svg.attr("width") - margin.left - margin.right,
	    height = +svg.attr("height") - margin.top - margin.bottom;
	svg.selectAll('g').remove();
	var g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");
	var x = d3.scaleTime()
	    .rangeRound([0, width]);
	var y = d3.scaleLinear()
	    .rangeRound([height, 0]);
	
	x.domain(d3.extent(values, function(d) { return d.time; }));  //extent: return min and max values
	y.domain(d3.extent(values, function(d) { return d.value; }));

	var line = d3.line()
		.x(function(d) { return x(d.time); })
	    .y(function(d) { return y(d.value); });
	
	//console.log('line chart', data, name)
	g.append("g")
	  	.call(d3.axisLeft(y)
		  .ticks(4)
          .tickSize(-width))
	  .append("text")
	  .attr("transform", "rotate(-90)")
	  .attr("y", 6)
	  .attr("dy", "0.71em")
	  .attr("text-anchor", "end");
	
	g.append("g")
	  .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(x)
    	  .ticks(3)
          .tickSize(-height));
	svg.selectAll(".tick line").attr("stroke", '#fff');
	svg.selectAll("path").attr("stroke", '#fff');
	svg.selectAll(".tick text").attr("fill", '#fff');
	
	
	var colors = ['Orange', 'Red', 'SteelBlue', 'LightCoral', 'VioletRed', 'Turquoise', 
		'Green', 'Yellow', 'Blue', 'SeaGreen', 'OliveDrab', 'LightSalmon' ];
	
	for(var i in data) {
		g.append("path")
		  .datum(data[i])
		  .attr("fill", "none")
		  .attr("stroke", colors[i])
		  .attr("stroke-linejoin", "round")
		  .attr("stroke-linecap", "round")
		  .attr("stroke-width", 1.5)
		  .attr("d", line);
		
		g.selectAll('dot')
			.data(data[i])
			.enter().append("circle")
			.attr("r", 2.5)
			.attr("fill", colors[i])
			.attr("cx", function(d) { return x(d.time); })
			.attr("cy", function(d) { return y(d.value); });
	}
	
};



