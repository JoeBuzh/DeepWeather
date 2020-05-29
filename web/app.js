var angular = require('angular'),
    ngroute = require('angular-route'),
    ngCookies = require('angular-cookies'),
    ngAnimate = require('angular-animate'),
    _ = require('lodash'),
    $ = require('jquery');
	//rzModuel = require('rzModule');

require('angular-ui-bootstrap');
require('angularjs-datepicker');
require('angularjs-slider');
require('ui-select');
//require('moment');

var app = angular.module('app', [
    'ngRoute',
    'ngCookies',
    'ngAnimate',
    'rzModule',
    //'720kb.datepicker',
    'ui.bootstrap',
    'ui.select'
    //'ngSanitize'
]);


require('./controllers/datePickerController');

require('./controllers/mapController');
require('./controllers/obsController');

require('./directives/barChartDirective');
require('./directives/lineChartDirective');


app.run(function($rootScope, $location, $cookies, $http, $window) {
    $rootScope.dt = new Date(2017, 6, 1);
});



