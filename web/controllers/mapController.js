'use strict';
var _ = require('lodash');
var moment = require('moment');

require('../services/olService');
require('../services/layerService');
var ol = require('openlayers');



angular
    .module('app')
    .controller('MapController', MapController);


function MapController(OlService, LayerService, $rootScope, $cookies, $routeParams, $scope, $http, $filter, $q, $compile) {
    var vm = this;
    init();
    function init(){
		$rootScope.selectTypes = [['Total precipitation', 0], 
				//['u-component of wind', 10],
				//['v-component of wind', 10],
				['Layer-maximum base reflectivity', 0],
				['wind', 10],
				['Global Radar', 0]];
		$rootScope.selectType = $scope.selectTypes[0];

		$scope.hour = moment().minute(0).second(0).milliseconds(0).toDate();
		
		setMap();
		setSlider();

		getLatLons().then(function(data){
			$rootScope.lats = data['lats'];
			$rootScope.lons = data['lons'];
			setRects();		

			getForecast();
		},function(error) {
			alert('err')
		});	

		
	}

	$scope.changeObsType = function($index) {
		//$rootScope.selectType = $rootScope.selectTypes[$index]
		$rootScope.map.removeLayer(OlService.rectLayer);
		setRects();
	}

	$scope.closePopup = function() {
    	var closer = document.getElementById('popup-closer');
    	var infoPop = $scope.map.getOverlayById('infoPop');
    	
    	infoPop.setPosition(undefined);
        closer.blur();
        return false;
	};
	
	function getForecast() {
		$rootScope.forecastValues = {
			values: {1: [
				{time: 1, value: 1},
				{time: 2, value: 2},
				{time: 3, value: 3}
			]},
			lenged: []
		};

		$rootScope.classValues = {
			values: [
				{type: 1, value: 0.8},
				{type: 1, value: 0.7},
				{type: 1, value: 0.6},
				{type: 1, value: 0.5},
				{type: 1, value: 0.8},
			],
			legend: ['沿海大风', '强对流大风', '暴雨', '雷电', '冰雹']
		}
	}
	
	function getLatLons() {
		var deferred = $q.defer();
		var promise = deferred.promise;
		$http({
    		method: 'GET',
    		url: '/rest/latlons'
    	}).then(function successCallback(response) {
			deferred.resolve(response.data);
    	}, function errorCallback(response) {
			deferred.reject(error);
		});
		return promise;
	}

	// add the weather data
	function setRects(isPredict) {
		if(isPredict===undefined) isPredict = 0;
		console.log($scope.selectType, isPredict);
		var time = $filter('date')($rootScope.dt, 'yyyyMMdd000000');
		$http({
    		method: 'GET',
    		url: '/rest/query?name='+$scope.selectType[0]+'&level='+$scope.selectType[1]+'&time='+time+'&ispredict='+isPredict
    	}).then(function successCallback(response) {
			LayerService.removeLayer($rootScope.map, 'rectLayer')
			$rootScope.data = response.data;
			var rectLayer = OlService.setRects($rootScope.map, response.data.values, $scope.selectType[0]);
			$rootScope.map.addLayer(rectLayer);

			var loc = OlService.addPopup($rootScope.map, response.data.values, [13123611.813590657, 3011244.213251752]);
			OlService.getPrediction(loc)
    	}, function errorCallback(response) {
    		
    	})
	}

    function setMap() {

		var pointLayer;
		var layers = [];
		var pointFeatures = [];
		var assetSource;
		var styles = ['img_w', 'cva_w', 'vec_w', 'cia_w'];

		var distance = document.getElementById('distance');

		var i, ii;
		var layers =[new ol.layer.Tile({
			visible: true,
			preload: Infinity,
			source: new ol.source.XYZ({
				// layers img_w, vec_w
				//url: 'http://www.arcgisonline.cn/arcgis/rest/services/ChinaOnlineCommunity/MapServer/tile/{z}/{y}/{x}'
				url: "http://webst0{1-4}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}"
			})
		})];
		$rootScope.map = new ol.Map({
			target: 'map',
			layers: layers,
			interactions: ol.interaction.defaults().extend([new OlService.Click()]),
			view: new ol.View({
				center: ol.proj.fromLonLat([118.067192, 25.950079]),
		        zoom: 7
			})
		});
		var infoPop = new ol.Overlay({
			id: 'infoPop',
			element: document.getElementById('popup')
		});
		$rootScope.map.addOverlay(infoPop);
		
		

	};
	
	function setSlider(date) {
    	if (!date){
    		var date = $rootScope.dt;
		} 
		//console.log(date.getTime());
    	//date = new Date(date.getTime());
    	var dates = [];
		for (var i = 0; i < 12 ; i++) {
		  dates.push(new Date(date.getTime()+i*60*60*1000));
		}
    	$scope.slider = {
		    value: dates[0],
		    options: {
		    	showTicks: 1,
		    	showTicksValues: 2,
			    stepsArray: dates,
			    translate: function(value) {
					return $filter('date')(value, 'MM-dd HH:00');
			    },
			    getTickColor: function (value) {
					if (value > 0){
						return 'lightblue';
					} else {
						return 'orange';
					}
			    	  	
		        },
			    onChange: function(id, value, highValue, pointerType) {
					var predictTime = value.getTime()
			    	console.log(predictTime);
			    }
			  }
			}
		}

}
