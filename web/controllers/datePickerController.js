angular
	.module('app')
	.controller('DatepickerCtrl', function ($scope, $rootScope) {
		$scope.today = function() {
	    //$rootScope.dt = new Date();
		};
		$scope.today();
	
		$scope.clear = function() {
			$rootScope.dt = null;
		};
	  
	  
	  
	
	  $scope.inlineOptions = {
	    customClass: getDayClass,
	    minDate: new Date(),
	    showWeeks: false,
	    
	  };
	
	  $scope.dateOptions = {
	    //dateDisabled: disabled,
	    formatYear: 'yy',
	    maxDate: new Date(2020, 5, 22),
	    minDate: new Date(),
	    startingDay: 1,
	    showWeeks: false
	  };
	
	  // Disable weekend selection
	  /*function disabled(data) {
	    var date = data.date,
	      mode = data.mode;
	    return mode === 'day' && (date.getDay() === 0 || date.getDay() === 6);
	  }*/
	
	  $scope.toggleMin = function() {
	    $scope.inlineOptions.minDate = $scope.inlineOptions.minDate ? null : new Date();
	    $scope.dateOptions.minDate = $scope.inlineOptions.minDate;
	  };
	
	  $scope.toggleMin();
	
	  /*$scope.open1 = function() {
	    $scope.popup1.opened = true;
	  };*/
	
	  $scope.open2 = function() {
	    $scope.popup2.opened = true;
	  };
	
	  $scope.setDate = function(year, month, day) {
	    $rootScope.dt = new Date(year, month, day);
	  };
	
	  $scope.formats = ['dd-MMMM-yyyy HH:mm:ss', 'yyyy/MM/dd', 'dd.MM.yyyy', 'shortDate'];
	  $scope.format = $scope.formats[0];
	  $scope.altInputFormats = ['M!/d!/yyyy'];
	
	  /*$scope.popup1 = {
	    opened: false
	  };*/
	
	  $scope.popup2 = {
	    opened: false
	  };
	
	  var tomorrow = new Date();
	  tomorrow.setDate(tomorrow.getDate() + 1);
	  var afterTomorrow = new Date();
	  afterTomorrow.setDate(tomorrow.getDate() + 1);
	  $scope.events = [
	    {
	      date: tomorrow,
	      status: 'full'
	    },
	    {
	      date: afterTomorrow,
	      status: 'partially'
	    }
	  ];
	
	  function getDayClass(data) {
	    var date = data.date,
	      mode = data.mode;
	    if (mode === 'day') {
	      var dayToCheck = new Date(date).setHours(0,0,0,0);
	
	      for (var i = 0; i < $scope.events.length; i++) {
	        var currentDay = new Date($scope.events[i].date).setHours(0,0,0,0);
	
	        if (dayToCheck === currentDay) {
	          return $scope.events[i].status;
	        }
	      }
	    }
	
	    return '';
	  }
});
