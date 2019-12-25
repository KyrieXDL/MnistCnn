"use strict";
function _classCallCheck(t,e){
    if(!(t instanceof e))
        throw new TypeError("Cannot call a class as a function")
    }
    var _createClass=function(){
        function t(t,e){
            for(var i=0;i<e.length;i++){
                var n=e[i];
                n.enumerable=n.enumerable||!1,n.configurable=!0,"value"in n&&(n.writable=!0),Object.defineProperty(t,n.key,n)
            }
        }
        return function(e,i,n){return i&&t(e.prototype,i),n&&t(e,n),e}
    }(), Main=function(){
    function t(){_classCallCheck(this,t),this.canvas=document.getElementById("main"),this.input=document.getElementById("input"),this.canvas.width=449,this.canvas.height=449,this.ctx=this.canvas.getContext("2d"),this.canvas.addEventListener("mousedown",this.onMouseDown.bind(this)),this.canvas.addEventListener("mouseup",this.onMouseUp.bind(this)),this.canvas.addEventListener("mousemove",this.onMouseMove.bind(this)),this.initialize()}return _createClass(t,[{key:"initialize",value:function(){this.ctx.fillStyle="#FFFFFF",this.ctx.fillRect(0,0,449,449),this.ctx.lineWidth=1,this.ctx.strokeRect(0,0,449,449),this.ctx.lineWidth=.05;for(var t=0;t<27;t++)this.ctx.beginPath(),this.ctx.moveTo(16*(t+1),0),this.ctx.lineTo(16*(t+1),449),this.ctx.closePath(),this.ctx.stroke(),this.ctx.beginPath(),this.ctx.moveTo(0,16*(t+1)),this.ctx.lineTo(449,16*(t+1)),this.ctx.closePath(),this.ctx.stroke();this.drawInput(),$("#output td").text("").removeClass("success")}},{key:"onMouseDown",value:function(t){this.canvas.style.cursor="default",this.drawing=!0,this.prev=this.getPosition(t.clientX,t.clientY)}},{key:"onMouseUp",value:function(){this.drawing=!1,this.drawInput()}},{key:"onMouseMove",value:function(t){if(this.drawing){var e=this.getPosition(t.clientX,t.clientY);this.ctx.lineWidth=16,this.ctx.lineCap="round",this.ctx.beginPath(),this.ctx.moveTo(this.prev.x,this.prev.y),this.ctx.lineTo(e.x,e.y),this.ctx.stroke(),this.ctx.closePath(),this.prev=e}}},{key:"getPosition",value:function(t,e){var i=this.canvas.getBoundingClientRect();return{x:t-i.left,y:e-i.top}}},{key:"drawInput",value:function(){var t=this.input.getContext("2d"),e=new Image;e.onload=function(){var i=[],n=document.createElement("canvas").getContext("2d");n.drawImage(e,0,0,e.width,e.height,0,0,28,28);for(var s=n.getImageData(0,0,28,28).data,a=0;a<28;a++)for(var o=0;o<28;o++){var c=4*(28*a+o);i[28*a+o]=(s[c+0]+s[c+1]+s[c+2])/3,t.fillStyle="rgb("+[s[c+0],s[c+1],s[c+2]].join(",")+")",t.fillRect(5*o,5*a,5,5)}255!==Math.min.apply(Math,i)&&$.ajax({url:"/api/mnist",method:"POST",contentType:"application/json",data:JSON.stringify(i),success:function(t){for(var e=0;e<2;e++){for(var i=0,n=0,s=0;s<10;s++){var a=Math.round(1e3*t.results[e][s]);a>i&&(i=a,n=s);for(var o=String(a).length,c=0;c<3-o;c++)a="0"+a;var r="0."+a;a>999&&(r="1.000"),$("#output tr").eq(s+1).find("td").eq(e).text(r)}for(var h=0;h<10;h++)h===n?$("#output tr").eq(h+1).find("td").eq(e).addClass("success"):$("#output tr").eq(h+1).find("td").eq(e).removeClass("success")}}})},e.src=this.canvas.toDataURL()}}]),t}();$(function(){var t=new Main;$("#clear").click(function(){t.initialize()})});
//# sourceMappingURL=data:application/json;charset=utf8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIm1haW4uanMiXSwibmFtZXMiOlsiX2NsYXNzQ2FsbENoZWNrIiwiaW5zdGFuY2UiLCJDb25zdHJ1Y3RvciIsIlR5cGVFcnJvciIsIl9jcmVhdGVDbGFzcyIsImRlZmluZVByb3BlcnRpZXMiLCJ0YXJnZXQiLCJwcm9wcyIsImkiLCJsZW5ndGgiLCJkZXNjcmlwdG9yIiwiZW51bWVyYWJsZSIsImNvbmZpZ3VyYWJsZSIsIndyaXRhYmxlIiwiT2JqZWN0IiwiZGVmaW5lUHJvcGVydHkiLCJrZXkiLCJwcm90b1Byb3BzIiwic3RhdGljUHJvcHMiLCJwcm90b3R5cGUiLCJNYWluIiwidGhpcyIsImNhbnZhcyIsImRvY3VtZW50IiwiZ2V0RWxlbWVudEJ5SWQiLCJpbnB1dCIsIndpZHRoIiwiaGVpZ2h0IiwiY3R4IiwiZ2V0Q29udGV4dCIsImFkZEV2ZW50TGlzdGVuZXIiLCJvbk1vdXNlRG93biIsImJpbmQiLCJvbk1vdXNlVXAiLCJvbk1vdXNlTW92ZSIsImluaXRpYWxpemUiLCJ2YWx1ZSIsImZpbGxTdHlsZSIsImZpbGxSZWN0IiwibGluZVdpZHRoIiwic3Ryb2tlUmVjdCIsImJlZ2luUGF0aCIsIm1vdmVUbyIsImxpbmVUbyIsImNsb3NlUGF0aCIsInN0cm9rZSIsImRyYXdJbnB1dCIsIiQiLCJ0ZXh0IiwicmVtb3ZlQ2xhc3MiLCJlIiwic3R5bGUiLCJjdXJzb3IiLCJkcmF3aW5nIiwicHJldiIsImdldFBvc2l0aW9uIiwiY2xpZW50WCIsImNsaWVudFkiLCJjdXJyIiwibGluZUNhcCIsIngiLCJ5IiwicmVjdCIsImdldEJvdW5kaW5nQ2xpZW50UmVjdCIsImxlZnQiLCJ0b3AiLCJpbWciLCJJbWFnZSIsIm9ubG9hZCIsImlucHV0cyIsInNtYWxsIiwiY3JlYXRlRWxlbWVudCIsImRyYXdJbWFnZSIsImRhdGEiLCJnZXRJbWFnZURhdGEiLCJqIiwibiIsImpvaW4iLCJNYXRoIiwibWluIiwiYXBwbHkiLCJhamF4IiwidXJsIiwibWV0aG9kIiwiY29udGVudFR5cGUiLCJKU09OIiwic3RyaW5naWZ5Iiwic3VjY2VzcyIsIl9pIiwibWF4IiwibWF4X2luZGV4IiwiX2oiLCJyb3VuZCIsInJlc3VsdHMiLCJkaWdpdHMiLCJTdHJpbmciLCJrIiwiZXEiLCJmaW5kIiwiX2oyIiwiYWRkQ2xhc3MiLCJzcmMiLCJ0b0RhdGFVUkwiLCJtYWluIiwiY2xpY2siXSwibWFwcGluZ3MiOiJBQUFBLFlBSUEsU0FBU0EsaUJBQWdCQyxFQUFVQyxHQUFlLEtBQU1ELFlBQW9CQyxJQUFnQixLQUFNLElBQUlDLFdBQVUscUNBRmhILEdBQUlDLGNBQWUsV0FBYyxRQUFTQyxHQUFpQkMsRUFBUUMsR0FBUyxJQUFLLEdBQUlDLEdBQUksRUFBR0EsRUFBSUQsRUFBTUUsT0FBUUQsSUFBSyxDQUFFLEdBQUlFLEdBQWFILEVBQU1DLEVBQUlFLEdBQVdDLFdBQWFELEVBQVdDLGFBQWMsRUFBT0QsRUFBV0UsY0FBZSxFQUFVLFNBQVdGLEtBQVlBLEVBQVdHLFVBQVcsR0FBTUMsT0FBT0MsZUFBZVQsRUFBUUksRUFBV00sSUFBS04sSUFBaUIsTUFBTyxVQUFVUixFQUFhZSxFQUFZQyxHQUFpSixNQUE5SEQsSUFBWVosRUFBaUJILEVBQVlpQixVQUFXRixHQUFpQkMsR0FBYWIsRUFBaUJILEVBQWFnQixHQUFxQmhCLE1BSzVoQmtCLEtBQU8sV0FDUCxRQUFTQSxLQUNMcEIsZ0JBQWdCcUIsS0FBTUQsR0FFdEJDLEtBQUtDLE9BQVNDLFNBQVNDLGVBQWUsUUFDdENILEtBQUtJLE1BQVFGLFNBQVNDLGVBQWUsU0FDckNILEtBQUtDLE9BQU9JLE1BQVEsSUFDcEJMLEtBQUtDLE9BQU9LLE9BQVMsSUFDckJOLEtBQUtPLElBQU1QLEtBQUtDLE9BQU9PLFdBQVcsTUFDbENSLEtBQUtDLE9BQU9RLGlCQUFpQixZQUFhVCxLQUFLVSxZQUFZQyxLQUFLWCxPQUNoRUEsS0FBS0MsT0FBT1EsaUJBQWlCLFVBQVdULEtBQUtZLFVBQVVELEtBQUtYLE9BQzVEQSxLQUFLQyxPQUFPUSxpQkFBaUIsWUFBYVQs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RUFBS3BFIiwiZmlsZSI6Im1haW4uanMiLCJzb3VyY2VzQ29udGVudCI6WyIndXNlIHN0cmljdCc7XG5cbnZhciBfY3JlYXRlQ2xhc3MgPSBmdW5jdGlvbiAoKSB7IGZ1bmN0aW9uIGRlZmluZVByb3BlcnRpZXModGFyZ2V0LCBwcm9wcykgeyBmb3IgKHZhciBpID0gMDsgaSA8IHByb3BzLmxlbmd0aDsgaSsrKSB7IHZhciBkZXNjcmlwdG9yID0gcHJvcHNbaV07IGRlc2NyaXB0b3IuZW51bWVyYWJsZSA9IGRlc2NyaXB0b3IuZW51bWVyYWJsZSB8fCBmYWxzZTsgZGVzY3JpcHRvci5jb25maWd1cmFibGUgPSB0cnVlOyBpZiAoXCJ2YWx1ZVwiIGluIGRlc2NyaXB0b3IpIGRlc2NyaXB0b3Iud3JpdGFibGUgPSB0cnVlOyBPYmplY3QuZGVmaW5lUHJvcGVydHkodGFyZ2V0LCBkZXNjcmlwdG9yLmtleSwgZGVzY3JpcHRvcik7IH0gfSByZXR1cm4gZnVuY3Rpb24gKENvbnN0cnVjdG9yLCBwcm90b1Byb3BzLCBzdGF0aWNQcm9wcykgeyBpZiAocHJvdG9Qcm9wcykgZGVmaW5lUHJvcGVydGllcyhDb25zdHJ1Y3Rvci5wcm90b3R5cGUsIHByb3RvUHJvcHMpOyBpZiAoc3RhdGljUHJvcHMpIGRlZmluZVByb3BlcnRpZXMoQ29uc3RydWN0b3IsIHN0YXRpY1Byb3BzKTsgcmV0dXJuIENvbnN0cnVjdG9yOyB9OyB9KCk7XG5cbmZ1bmN0aW9uIF9jbGFzc0NhbGxDaGVjayhpbnN0YW5jZSwgQ29uc3RydWN0b3IpIHsgaWYgKCEoaW5zdGFuY2UgaW5zdGFuY2VvZiBDb25zdHJ1Y3RvcikpIHsgdGhyb3cgbmV3IFR5cGVFcnJvcihcIkNhbm5vdCBjYWxsIGEgY2xhc3MgYXMgYSBmdW5jdGlvblwiKTsgfSB9XG5cbi8qIGdsb2JhbCAkICovXG52YXIgTWFpbiA9IGZ1bmN0aW9uICgpIHtcbiAgICBmdW5jdGlvbiBNYWluKCkge1xuICAgICAgICBfY2xhc3NDYWxsQ2hlY2sodGhpcywgTWFpbik7XG5cbiAgICAgICAgdGhpcy5jYW52YXMgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbWFpbicpO1xuICAgICAgICB0aGlzLmlucHV0ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2lucHV0Jyk7XG4gICAgICAgIHRoaXMuY2FudmFzLndpZHRoID0gNDQ5OyAvLyAxNiAqIDI4ICsgMVxuICAgICAgICB0aGlzLmNhbnZhcy5oZWlnaHQgPSA0NDk7IC8vIDE2ICogMjggKyAxXG4gICAgICAgIHRoaXMuY3R4ID0gdGhpcy5jYW52YXMuZ2V0Q29udGV4dCgnMmQnKTtcbiAgICAgICAgdGhpcy5jYW52YXMuYWRkRXZlbnRMaXN0ZW5lcignbW91c2Vkb3duJywgdGhpcy5vbk1vdXNlRG93bi5iaW5kKHRoaXMpKTtcbiAgICAgICAgdGhpcy5jYW52YXMuYWRkRXZlbnRMaXN0ZW5lcignbW91c2V1cCcsIHRoaXMub25Nb3VzZVVwLmJpbmQodGhpcykpO1xuICAgICAgICB0aGlzLmNhbnZhcy5hZGRFdmVudExpc3RlbmVyKCdtb3VzZW1vdmUnLCB0aGlzLm9uTW91c2VNb3ZlLmJpbmQodGhpcykpO1xuICAgICAgICB0aGlzLmluaXRpYWxpemUoKTtcbiAgICB9XG5cbiAgICBfY3JlYXRlQ2xhc3MoTWFpbiwgW3tcbiAgICAgICAga2V5OiAnaW5pdGlhbGl6ZScsXG4gICAgICAgIHZhbHVlOiBmdW5jdGlvbiBpbml0aWFsaXplKCkge1xuICAgICAgICAgICAgdGhpcy5jdHguZmlsbFN0eWxlID0gJyNGRkZGRkYnO1xuICAgICAgICAgICAgdGhpcy5jdHguZmlsbFJlY3QoMCwgMCwgNDQ5LCA0NDkpO1xuICAgICAgICAgICAgdGhpcy5jdHgubGluZVdpZHRoID0gMTtcbiAgICAgICAgICAgIHRoaXMuY3R4LnN0cm9rZVJlY3QoMCwgMCwgNDQ5LCA0NDkpO1xuICAgICAgICAgICAgdGhpcy5jdHgubGluZVdpZHRoID0gMC4wNTtcbiAgICAgICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgMjc7IGkrKykge1xuICAgICAgICAgICAgICAgIHRoaXMuY3R4LmJlZ2luUGF0aCgpO1xuICAgICAgICAgICAgICAgIHRoaXMuY3R4Lm1vdmVUbygoaSArIDEpICogMTYsIDApO1xuICAgICAgICAgICAgICAgIHRoaXMuY3R4LmxpbmVUbygoaSArIDEpICogMTYsIDQ0OSk7XG4gICAgICAgICAgICAgICAgdGhpcy5jdHguY2xvc2VQYXRoKCk7XG4gICAgICAgICAgICAgICAgdGhpcy5jdHguc3Ryb2tlKCk7XG5cbiAgICAgICAgICAgICAgICB0aGlzLmN0eC5iZWdpblBhdGgoKTtcbiAgICAgICAgICAgICAgICB0aGlzLmN0eC5tb3ZlVG8oMCwgKGkgKyAxKSAqIDE2KTtcbiAgICAgICAgICAgICAgICB0aGlzLmN0eC5saW5lVG8oNDQ5LCAoaSArIDEpICogMTYpO1xuICAgICAgICAgICAgICAgIHRoaXMuY3R4LmNsb3NlUGF0aCgpO1xuICAgICAgICAgICAgICAgIHRoaXMuY3R4LnN0cm9rZSgpO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgdGhpcy5kcmF3SW5wdXQoKTtcbiAgICAgICAgICAgICQoJyNvdXRwdXQgdGQnKS50ZXh0KCcnKS5yZW1vdmVDbGFzcygnc3VjY2VzcycpO1xuICAgICAgICB9XG4gICAgfSwge1xuICAgICAgICBrZXk6ICdvbk1vdXNlRG93bicsXG4gICAgICAgIHZhbHVlOiBmdW5jdGlvbiBvbk1vdXNlRG93bihlKSB7XG4gICAgICAgICAgICB0aGlzLmNhbnZhcy5zdHlsZS5jdXJzb3IgPSAnZGVmYXVsdCc7XG4gICAgICAgICAgICB0aGlzLmRyYXdpbmcgPSB0cnVlO1xuICAgICAgICAgICAgdGhpcy5wcmV2ID0gdGhpcy5nZXRQb3NpdGlvbihlLmNsaWVudFgsIGUuY2xpZW50WSk7XG4gICAgICAgIH1cbiAgICB9LCB7XG4gICAgICAgIGtleTogJ29uTW91c2VVcCcsXG4gICAgICAgIHZhbHVlOiBmdW5jdGlvbiBvbk1vdXNlVXAoKSB7XG4gICAgICAgICAgICB0aGlzLmRyYXdpbmcgPSBmYWxzZTtcbiAgICAgICAgICAgIHRoaXMuZHJhd0lucHV0KCk7XG4gICAgICAgIH1cbiAgICB9LCB7XG4gICAgICAgIGtleTogJ29uTW91c2VNb3ZlJyxcbiAgICAgICAgdmFsdWU6IGZ1bmN0aW9uIG9uTW91c2VNb3ZlKGUpIHtcbiAgICAgICAgICAgIGlmICh0aGlzLmRyYXdpbmcpIHtcbiAgICAgICAgICAgICAgICB2YXIgY3VyciA9IHRoaXMuZ2V0UG9zaXRpb24oZS5jbGllbnRYLCBlLmNsaWVudFkpO1xuICAgICAgICAgICAgICAgIHRoaXMuY3R4LmxpbmVXaWR0aCA9IDE2O1xuICAgICAgICAgICAgICAgIHRoaXMuY3R4LmxpbmVDYXAgPSAncm91bmQnO1xuICAgICAgICAgICAgICAgIHRoaXMuY3R4LmJlZ2luUGF0aCgpO1xuICAgICAgICAgICAgICAgIHRoaXMuY3R4Lm1vdmVUbyh0aGlzLnByZXYueCwgdGhpcy5wcmV2LnkpO1xuICAgICAgICAgICAgICAgIHRoaXMuY3R4LmxpbmVUbyhjdXJyLngsIGN1cnIueSk7XG4gICAgICAgICAgICAgICAgdGhpcy5jdHguc3Ryb2tlKCk7XG4gICAgICAgICAgICAgICAgdGhpcy5jdHguY2xvc2VQYXRoKCk7XG4gICAgICAgICAgICAgICAgdGhpcy5wcmV2ID0gY3VycjtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgIH0sIHtcbiAgICAgICAga2V5OiAnZ2V0UG9zaXRpb24nLFxuICAgICAgICB2YWx1ZTogZnVuY3Rpb24gZ2V0UG9zaXRpb24oY2xpZW50WCwgY2xpZW50WSkge1xuICAgICAgICAgICAgdmFyIHJlY3QgPSB0aGlzLmNhbnZhcy5nZXRCb3VuZGluZ0NsaWVudFJlY3QoKTtcbiAgICAgICAgICAgIHJldHVybiB7XG4gICAgICAgICAgICAgICAgeDogY2xpZW50WCAtIHJlY3QubGVmdCxcbiAgICAgICAgICAgICAgICB5OiBjbGllbnRZIC0gcmVjdC50b3BcbiAgICAgICAgICAgIH07XG4gICAgICAgIH1cbiAgICB9LCB7XG4gICAgICAgIGtleTogJ2RyYXdJbnB1dCcsXG4gICAgICAgIHZhbHVlOiBmdW5jdGlvbiBkcmF3SW5wdXQoKSB7XG4gICAgICAgICAgICB2YXIgY3R4ID0gdGhpcy5pbnB1dC5nZXRDb250ZXh0KCcyZCcpO1xuICAgICAgICAgICAgdmFyIGltZyA9IG5ldyBJbWFnZSgpO1xuICAgICAgICAgICAgaW1nLm9ubG9hZCA9IGZ1bmN0aW9uICgpIHtcbiAgICAgICAgICAgICAgICB2YXIgaW5wdXRzID0gW107XG4gICAgICAgICAgICAgICAgdmFyIHNtYWxsID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnY2FudmFzJykuZ2V0Q29udGV4dCgnMmQnKTtcbiAgICAgICAgICAgICAgICBzbWFsbC5kcmF3SW1hZ2UoaW1nLCAwLCAwLCBpbWcud2lkdGgsIGltZy5oZWlnaHQsIDAsIDAsIDI4LCAyOCk7XG4gICAgICAgICAgICAgICAgdmFyIGRhdGEgPSBzbWFsbC5nZXRJbWFnZURhdGEoMCwgMCwgMjgsIDI4KS5kYXRhO1xuICAgICAgICAgICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgMjg7IGkrKykge1xuICAgICAgICAgICAgICAgICAgICBmb3IgKHZhciBqID0gMDsgaiA8IDI4OyBqKyspIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHZhciBuID0gNCAqIChpICogMjggKyBqKTtcbiAgICAgICAgICAgICAgICAgICAgICAgIGlucHV0c1tpICogMjggKyBqXSA9IChkYXRhW24gKyAwXSArIGRhdGFbbiArIDFdICsgZGF0YVtuICsgMl0pIC8gMztcbiAgICAgICAgICAgICAgICAgICAgICAgIGN0eC5maWxsU3R5bGUgPSAncmdiKCcgKyBbZGF0YVtuICsgMF0sIGRhdGFbbiArIDFdLCBkYXRhW24gKyAyXV0uam9pbignLCcpICsgJyknO1xuICAgICAgICAgICAgICAgICAgICAgICAgY3R4LmZpbGxSZWN0KGogKiA1LCBpICogNSwgNSwgNSk7XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgaWYgKE1hdGgubWluLmFwcGx5KE1hdGgsIGlucHV0cykgPT09IDI1NSkge1xuICAgICAgICAgICAgICAgICAgICByZXR1cm47XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICQuYWpheCh7XG4gICAgICAgICAgICAgICAgICAgIHVybDogJy9hcGkvbW5pc3QnLFxuICAgICAgICAgICAgICAgICAgICBtZXRob2Q6ICdQT1NUJyxcbiAgICAgICAgICAgICAgICAgICAgY29udGVudFR5cGU6ICdhcHBsaWNhdGlvbi9qc29uJyxcbiAgICAgICAgICAgICAgICAgICAgZGF0YTogSlNPTi5zdHJpbmdpZnkoaW5wdXRzKSxcbiAgICAgICAgICAgICAgICAgICAgc3VjY2VzczogZnVuY3Rpb24gc3VjY2VzcyhkYXRhKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICBmb3IgKHZhciBfaSA9IDA7IF9pIDwgMjsgX2krKykge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHZhciBtYXggPSAwO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHZhciBtYXhfaW5kZXggPSAwO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGZvciAodmFyIF9qID0gMDsgX2ogPCAxMDsgX2orKykge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB2YXIgdmFsdWUgPSBNYXRoLnJvdW5kKGRhdGEucmVzdWx0c1tfaV1bX2pdICogMTAwMCk7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmICh2YWx1ZSA+IG1heCkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbWF4ID0gdmFsdWU7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBtYXhfaW5kZXggPSBfajtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB2YXIgZGlnaXRzID0gU3RyaW5nKHZhbHVlKS5sZW5ndGg7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGZvciAodmFyIGsgPSAwOyBrIDwgMyAtIGRpZ2l0czsgaysrKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB2YWx1ZSA9ICcwJyArIHZhbHVlO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHZhciB0ZXh0ID0gJzAuJyArIHZhbHVlO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAodmFsdWUgPiA5OTkpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRleHQgPSAnMS4wMDAnO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICQoJyNvdXRwdXQgdHInKS5lcShfaiArIDEpLmZpbmQoJ3RkJykuZXEoX2kpLnRleHQodGV4dCk7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGZvciAodmFyIF9qMiA9IDA7IF9qMiA8IDEwOyBfajIrKykge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAoX2oyID09PSBtYXhfaW5kZXgpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICQoJyNvdXRwdXQgdHInKS5lcShfajIgKyAxKS5maW5kKCd0ZCcpLmVxKF9pKS5hZGRDbGFzcygnc3VjY2VzcycpO1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJCgnI291dHB1dCB0cicpLmVxKF9qMiArIDEpLmZpbmQoJ3RkJykuZXEoX2kpLnJlbW92ZUNsYXNzKCdzdWNjZXNzJyk7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB9KTtcbiAgICAgICAgICAgIH07XG4gICAgICAgICAgICBpbWcuc3JjID0gdGhpcy5jYW52YXMudG9EYXRhVVJMKCk7XG4gICAgICAgIH1cbiAgICB9XSk7XG5cbiAgICByZXR1cm4gTWFpbjtcbn0oKTtcblxuJChmdW5jdGlvbiAoKSB7XG4gICAgdmFyIG1haW4gPSBuZXcgTWFpbigpO1xuICAgICQoJyNjbGVhcicpLmNsaWNrKGZ1bmN0aW9uICgpIHtcbiAgICAgICAgbWFpbi5pbml0aWFsaXplKCk7XG4gICAgfSk7XG59KTsiXX0=