'use strict';

var _ = require('lodash');
var ol = require('openlayers');
var $ = require('jquery');

angular
    .module('app')
    .factory('OlService', OlService);

//OlService.$inject = ['$http', '$q', '$filter'];

function OlService($http, $q, $filter, $rootScope) {
	var service = {};

	service.getRtLb = function(data) {
		service.lb = [$rootScope.lons[0][0], $rootScope.lats[0][0]];
		service.rt = [$rootScope.lons[0][$rootScope.lons.length-1], $rootScope.lats[$rootScope.lats.length-1][0]];
	}
	
	service.setRects = function(map, values, type) {
		var dataWidth = values.length, dataHeight = values[0].length
		var canvasFunction=function(extent, resolution, pixelRatio, size, projection){  
			var width = $('#map').width()*pixelRatio;
			var height = $('#map').height()*pixelRatio;

			var canvas = document.createElement('canvas');
			var context = canvas.getContext('2d');
			canvas.setAttribute('width', size[0]);
			canvas.setAttribute('height', size[1]);

			var mapExtent = map.getView().calculateExtent(map.getSize())
			var canvasOrigin = map.getPixelFromCoordinate([extent[0], extent[3]]);
			var mapOrigin = map.getPixelFromCoordinate([mapExtent[0], mapExtent[3]]);
			var delta = [(mapOrigin[0]-canvasOrigin[0])*pixelRatio, (mapOrigin[1]-canvasOrigin[1])*pixelRatio];


			var rt = {'coord': ol.proj.fromLonLat(service.rt)};
			var lb = {'coord': ol.proj.fromLonLat(service.lb)};
			rt.pixel = map.getPixelFromCoordinate(rt.coord);
			lb.pixel = map.getPixelFromCoordinate(lb.coord);
			rt.pixel = [rt.pixel[0], rt.pixel[1]];
			lb.pixel = [lb.pixel[0], lb.pixel[1]];

			
			var columnStep = (rt.pixel[0]-lb.pixel[0])*pixelRatio/dataWidth;  // when zoom in, use step to improve the performance
			var rowStep = (lb.pixel[1] - rt.pixel[1])*pixelRatio/dataHeight;
			columnStep = columnStep >= 1 ? columnStep : 1;  // improve performance when zoom out
			rowStep = rowStep >= 1? rowStep: 1;

			console.log(columnStep, rowStep, type)
			for(var left = Math.max(0, lb.pixel[0]*pixelRatio); left < Math.min(rt.pixel[0]*pixelRatio, width); left+=columnStep){
				for(var top = Math.max(0, rt.pixel[1]*pixelRatio); top < Math.min(lb.pixel[1]*pixelRatio, height); top+=rowStep){
					var row = Math.round((lb.pixel[1]-top/pixelRatio)/(lb.pixel[1] - rt.pixel[1])*(dataWidth-1));
					var col = Math.round((left/pixelRatio-lb.pixel[0])/(rt.pixel[0]-lb.pixel[0])*(dataHeight-1));
					
					var value = values[row][col];
					context.fillStyle = getShade(value, type);

					// TODO recalculate the rect width and height when at the edge
					//var rectWidth = 
					//var rectHeight = 0;

					context.fillRect(delta[0]+left, delta[1]+top, Math.ceil(columnStep), Math.ceil(rowStep));
				}
			}
			return canvas;  
		};  
		this.getRtLb();
		var rectLayer = new ol.layer.Image({
			source: new ol.source.ImageCanvas({
				canvasFunction: canvasFunction,
			})
		});
		rectLayer.set('name', 'rectLayer');
		return rectLayer;
	}

	service.Click = function() {
	    ol.interaction.Pointer.call(this, {
	        handleDownEvent: service.Click.prototype.handleDownEvent
	    });
	
	    this.coordinate_ = null;
	    this.cursor_ = 'pointer';
	    this.feature_ = null;
	    this.previousCursor_ = undefined;
	};
	ol.inherits(service.Click, ol.interaction.Pointer);
	
	service.Click.prototype.handleDownEvent = function(evt) {
		if($rootScope.data === undefined){
			return;
		}
		var map = evt.map;

		var loc = service.addPopup(map, $rootScope.data.values, evt.coordinate);

		service.getPrediction(loc)

	};

	service.getPrediction = function(loc){
		var time = $filter('date')($rootScope.dt, 'yyyyMMdd000000');
		$http({
    		method: 'GET',
    		url: '/rest/prediction?name='+$rootScope.selectType[0]+'&level='+$rootScope.selectType[1]+'&time='+time+'&row='+loc[0]+'&col='+loc[1]
    	}).then(function successCallback(response) {
			var data = response.data;
			console.log('prediction', response.data)

			$rootScope.forecastValues = {
				values: {1: [
					{time: 1, value: data['value'][0]},
					{time: 2, value: data['value'][1]},
					{time: 3, value: data['value'][2]}
				]},
				lenged: []
			};
			$rootScope.classValues = {
				values: [
					{type: 1, value: data['prob'][0]},
					{type: 1, value: data['prob'][1]},
					{type: 1, value: data['prob'][2]},
					{type: 1, value: data['prob'][3]},
					{type: 1, value: data['prob'][4]},
				],
				legend: ['沿海大风', '强对流大风', '暴雨', '雷电', '冰雹']
			}
    	}, function errorCallback(response) {
    		
    	})
	}

	service.addPopup = function(map, values, coordinate){
		var width = values.length, height = values[0].length;

		var pixel = map.getPixelFromCoordinate(coordinate)
		
		var rt = {'coord': ol.proj.fromLonLat(service.rt)};
		var lb = {'coord': ol.proj.fromLonLat(service.lb)};
		rt.pixel = map.getPixelFromCoordinate(rt.coord);
		lb.pixel = map.getPixelFromCoordinate(lb.coord);
		rt.pixel = [rt.pixel[0], rt.pixel[1]];
		lb.pixel = [lb.pixel[0], lb.pixel[1]];

		var row = Math.round((lb.pixel[1]-pixel[1])/(lb.pixel[1] - rt.pixel[1])*(width-1));
		var col = Math.round((pixel[0]-lb.pixel[0])/(rt.pixel[0]-lb.pixel[0])*(height-1));

		if(row >= 0 && row < height && col >= 0 && col < width){
			//console.log(row+' '+col)

			var infoPop = map.getOverlayById('infoPop');
			//xvar element = infoPop.getElement();
			var latlon = ol.proj.transform(coordinate, 'EPSG:3857', 'EPSG:4326')
			//var hdms = ol.coordinate.toStringHDMS(latlon);
			var content = document.getElementById('popup-content');
			content.innerHTML = 
				"位置："+ latlon[0].toFixed(2)+", "+latlon[1].toFixed(2) +"<br>"+
				"天气状况："+$rootScope.data.values[row][col]+"<br>"
				"预报：";
			//console.log(row, col, pixel, coordinate);
			infoPop.setPosition(coordinate);  
		}
		return [row, col]
	}

	
    
    /*service.setSites = function(sites) {
    	
    	//TODO water level and value
    	//service.site = sites[0];
    	var siteFeatures = [];
    	sites.map(function(site) {
    		var point = new ol.geom.Point(ol.proj.fromLonLat([site.LONGITUDE, site.LATITUDE]));
			var pointFeature = new ol.Feature({
		        id_: site.ID,
		        geometry: point,
		        status: 'good'
		    });
			pointFeature.setId(site.ID);
			pointFeature.set('name', 'site');
			pointFeature.set('site', site);
			siteFeatures.push(pointFeature);
    	});
    	var siteSource = new ol.source.Vector({
			features: siteFeatures
		})
		var siteLayer = new ol.layer.Vector({
			source: siteSource,
			style: assetStyle,
			//zIndex: 10
		});
    	siteLayer.set('name', 'siteLayer');
		//map.addLayer(siteLayer);
    	service.siteLayer = siteLayer;
    }*/
    return service;
}

function getShade(value, type) {
	var color = 'rgba(255, 255, 255, 0.1)'
	if(type == 'Total precipitation'){
		if(value >= 0.1 && value < 0.5){
			color = 'rgba(102, 255, 255, 0.9)'
		} else if(value >=0.5 && value < 1.0){
			color = 'rgba(0, 119, 153, 0.9)'
		} else if(value >=1.0 && value < 2.0){
			color = 'rgba(0, 136, 0, 0.9)'
		} else if(value >=2.0 && value < 3.0){
			color = 'rgba(0, 255, 0, 0.9)'
		} else if(value >=3.0 && value < 4.0){
			color = 'rgba(255, 255, 0, 0.9)'
		} else if(value >=4.0 && value < 5.0){
			color = 'rgba(255, 187, 0, 0.9)'
		} else if(value >=5.0 && value < 6.0){
			color = 'rgba(136, 102, 255, 0.9)'
		} else if(value >=6.0 && value < 8.0){
			color = 'rgba(136, 0, 0, 0.9)'
		} else if(value >=8.0 && value < 10.0){
			color = 'rgba(204, 0, 0, 0.9)'
		} else if(value >=10.0 && value < 20.0){
			color = 'rgba(255, 68, 170, 0.9)'
		} else if(value >=20.0 && value < 40.0){
			color = 'rgba(165, 0, 204, 0.9)'
		} else if(value >=40.0){
			color = 'rgba(85, 0, 136, 0.9)'
		} 
		

	} else if(type == 'Layer-maximum base reflectivity' || type == 'Global Radar'){
		if(value >= 10 && value < 15){
			color = 'rgba(0, 187, 255, 0.9)'
		} else if(value >=15 && value < 20){
			color = 'rgba(0, 255, 255, 0.9)'
		} else if(value >=20 && value < 25){
			color = 'rgba(0, 221, 0, 0.9)'
		} else if(value >=25 && value < 30){
			color = 'rgba(34, 119, 0, 0.9)'
		} else if(value >=30 && value < 35){
			color = 'rgba(255, 255, 0, 0.9)'
		} else if(value >=35 && value < 40){
			color = 'rgba(255, 187, 85, 0.9)'
		} else if(value >=40 && value < 45){
			color = 'rgba(255, 136, 0, 0.9)'
		} else if(value >=45 && value < 50){
			color = 'rgba(255, 0, 0, 0.9)'
		} else if(value >=50 && value < 55){
			color = 'rgba(204, 0, 0, 0.9)'
		} else if(value >=55 && value < 60){
			color = 'rgba(170, 0, 0, 0.9)'
		} else if(value >=60 && value < 65){
			color = 'rgba(255, 0, 255, 0.9)'
		} else if(value >=65 && value < 70){
			color = 'rgba(122, 0, 153, 0.9)'
		} else if(value > 70){
			color = 'rgba(159, 136, 255, 0.9)'
		}
	} else if(type == 'wind'){
		if(value >=0 && value < 2.5 ){
			color = 'rgba(0, 187, 255, 0.9)'
		} else if(value >= 2.5 && value < 5){
			color = 'rgba(51, 255, 255, 0.9)'
		} else if(value >=5 && value < 7.5){
			color = 'rgba(0, 255, 255, 0.9)'
		} else if(value >=7.5 && value < 10){
			color = 'rgba(187, 255, 102, 0.9)'
		} else if(value >=10 && value < 12.5){
			color = 'rgba(255, 0, 0, 0.9)'
		} else if(value >=12.5 && value < 15){
			color = 'rgba(255, 106, 34, 0.9)'
		} else if(value >=15 && value < 17.5){
			color = 'rgba(239, 119, 0, 0.9)'
		} else if(value >=17.5 && value < 20){
			color = 'rgba(255, 0, 0, 0.9)'
		} else if(value >=20 && value < 22.5){
			color = 'rgba(206, 0, 206, 0.9)'
		} else if(value >=22.5 && value < 25){
			color = 'rgba(240, 187, 255, 0.9)'
		} else if(value >=25 && value < 27.5){
			color = 'rgba(255, 255, 255, 0.9)'
		} else if(value >=27.5){
			color = 'rgba(0, 0, 171, 0.9)'
		}
	}
	return color;

}

function gridStyle(feature) {
	var color = 'rgba(0, 205, 0, 0.8)';
	
	var style = new ol.style.Style({
	  fill: new ol.style.Fill({
		color: color
	  }),
	  radius: 8
	});
	return [style];
}
