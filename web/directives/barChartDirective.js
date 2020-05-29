'use strict';

var d3 = require('d3');
var $ = require('jquery');

angular.module('app').
   directive('barChart', BarChartDirective)
   
function BarChartDirective($parse) {
	return {
        scope: {
            values: '='
        },
        template: '<svg width="390" height="205"></svg>',
        restrict: 'E',
        controller: () => {},
        bindToController: true,
        controllerAs: 'viz',
        link: function (scope, element, attrs, ctrl) {
            // Bring in the Bubbles class
            // Create a Bubbles visualization, targeting the SVG element from the template
            var visualization = new BarChart(element.find('svg')[0]);
            // Watch for any changes to the values array, and when it changes, re-render the chart
            scope.$watchCollection(function () {
				return ctrl.values;
			}, function () {
				//console.log(attrs.name+' 12');
				visualization.render(ctrl.values ? ctrl.values : [], attrs.name);
			});
            scope.$on('$destroy', () => {
                // If we have anything to clean up when the scope gets destroyed
                visualization.destroy();
            });
        }
    };
}



   
function BarChart(target) {
    this.target = target;
}

function type(d, _, columns) {
    d.time = parseTime(d.time);
    for (var i = 1, n = columns.length, c; i < n; ++i) d[c = columns[i]] = +d[c];
    return d;
}
var parseTime = d3.timeParse("%Y-%m-%d %H:%M:%S");
 
// Does the work of drawing the visualization in the target area
BarChart.prototype.render = function (data, name) {
	if (data.values === undefined) {
		return;
	}

	var values = data.values.map(function(d, i){
		return {
			name: data.legend[i],
			value: +d.value
		}
	});
	console.log('bar chart ', values)
	
	var svg = d3.select(this.target);
	var margin = {top: 20, right: 20, bottom: 20, left: 30},
	    width = +svg.attr("width") - margin.left - margin.right,
	    height = +svg.attr("height") - margin.top - margin.bottom;
	svg.selectAll('g').remove();

	var g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

	var x = d3.scaleBand()
			.range([0, width])
			.padding(0.5);
	var y = d3.scaleLinear()
			.range([height, 0]);
			
	// append the svg object to the body of the page
	// append a 'group' element to 'svg'
	// moves the 'group' element to the top left margin

	// Scale the range of the data in the domains
	x.domain(values.map(function(d) { return d.name; }));
	y.domain([0, 1.0]);

	// append the rectangles for the bar chart
	g.selectAll(".bar")
		.data(values)
		.enter().append("rect")
		.attr("class", "bar")
		.attr("x", function(d) { return x(d.name); })
		.attr("width", x.bandwidth())
		.attr("y", function(d) { 
			//console.log(y(d.value));
			return y(d.value); })
		.attr("height", function(d) { 
			//console.log(height-y(d.value))
			return height - y(d.value); });

	// add the x Axis
	g.append("g")
		.attr("transform", "translate(0," + height + ")")
		.call(d3.axisBottom(x));

	// add the y Axis
	g.append("g")
		.call(d3.axisLeft(y));
	
		svg.selectAll(".tick line").attr("stroke", '#fff');
		svg.selectAll("path").attr("stroke", '#fff');
		svg.selectAll(".tick text").attr("fill", '#fff');
	
}
	



